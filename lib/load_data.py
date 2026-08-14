import numpy as np

from .load_llff import load_llff_data
from .load_blender import load_blender_data
from .load_nsvf import load_nsvf_data
from .load_blendedmvs import load_blendedmvs_data
from .load_tankstemple import load_tankstemple_data
from .load_deepvoxels import load_dv_data
from .load_co3d import load_co3d_data
from .load_nerfpp import load_nerfpp_data
from .load_replica import load_replica_data
from .load_lerf import load_lerf_data

# 카메라 거리 계산 가중치
# alpha: 회전 거리 가중치, beta: history 가중치
# alpha=1.0 : 카메라가 1 rad 회전시 시점이 1 m 이동했을 때
_GEOMETRY_POSE_ALPHA = 1.0
_GEOMETRY_HISTORY_BETA = 0.5

# 카메라 거리가 같으면 카메라 ID가 작은 것을 우선으로 선택
def _stable_argmin(values, candidate_indices, camera_ids):
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_values = np.asarray(values)[candidate_indices]
    minimum = np.min(candidate_values)
    tied = candidate_indices[
        np.isclose(candidate_values, minimum, rtol=1e-10, atol=1e-12)
    ]
    return int(tied[np.argmin(np.asarray(camera_ids)[tied])])

# 평균 pose 계산
def _compute_mean_pose(poses):
    rotations = poses[:, :3, :3]
    mean_translation = poses[:, :3, 3].mean(axis=0)
    mean_rotation_matrix = rotations.mean(axis=0)

    u, _, vt = np.linalg.svd(mean_rotation_matrix)
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0

    mean_pose = np.eye(4, dtype=np.float64)
    mean_pose[:3, :3] = u @ correction @ vt
    mean_pose[:3, 3] = mean_translation
    return mean_pose

# 카메라 각도 차이 계산
def _rotation_distance(rotation_a, rotation_b):
    relative_rotation = rotation_a.T @ rotation_b
    cosine = (np.trace(relative_rotation) - 1.0) / 2.0
    return np.arccos(np.clip(cosine, -1.0, 1.0))

# 카메라 pose 거리 계산
def _compute_pose_distance(pose_a, pose_b, alpha=_GEOMETRY_POSE_ALPHA):
    rotation_distance = _rotation_distance(
        pose_a[:3, :3],
        pose_b[:3, :3],
    )
    translation_distance = np.linalg.norm(
        pose_a[:3, 3] - pose_b[:3, 3]
    )
    return rotation_distance + alpha * translation_distance

# 전체 카메라 ordering
def _build_geometry_order(
        poses,
        camera_ids,
        alpha=_GEOMETRY_POSE_ALPHA,
        beta=_GEOMETRY_HISTORY_BETA):
    poses = np.asarray(poses, dtype=np.float64)
    camera_ids = np.asarray(camera_ids, dtype=np.int64)

    if poses.ndim != 3 or poses.shape[1] < 3 or poses.shape[2] < 4:
        raise ValueError("poses must have shape [N, 3, 4] or [N, 4, 4]")
    if len(poses) == 0 or len(poses) != len(camera_ids):
        raise ValueError("poses and camera_ids must contain the same cameras")
    if not np.all(np.isfinite(poses)):
        raise ValueError("poses must contain only finite values")
    if len(np.unique(camera_ids)) != len(camera_ids):
        raise ValueError("camera_ids must be unique")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if not np.isfinite(beta) or not 0.0 < beta < 1.0:
        raise ValueError("beta must satisfy 0 < beta < 1")

    # mean pose에 가장 가까운 카메라를 시작점으로 선택
    mean_pose = _compute_mean_pose(poses)
    mean_distances = np.asarray([
        _compute_pose_distance(pose, mean_pose, alpha)
        for pose in poses
    ])
    start_index = _stable_argmin(
        mean_distances,
        np.arange(len(camera_ids)),
        camera_ids,
    )

    distances = np.zeros((len(camera_ids), len(camera_ids)), dtype=np.float64)
    for first in range(len(camera_ids)):
        for second in range(first + 1, len(camera_ids)):
            distance = _compute_pose_distance(
                poses[first],
                poses[second],
                alpha,
            )
            distances[first, second] = distance
            distances[second, first] = distance

    remaining = np.ones(len(camera_ids), dtype=bool)
    remaining[start_index] = False
    ordered_indices = [start_index]
    weighted_scores = distances[:, start_index].copy()

    while remaining.any():
        candidate_indices = np.flatnonzero(remaining)
        next_index = _stable_argmin(
            weighted_scores,
            candidate_indices,
            camera_ids,
        )
        ordered_indices.append(next_index)
        remaining[next_index] = False

        # History 가중치 적용
        if remaining.any():
            weighted_scores = (
                beta * weighted_scores + distances[:, next_index]
            )

    ordered_camera_ids = camera_ids[ordered_indices]
    if len(np.unique(ordered_camera_ids)) != len(camera_ids):
        raise RuntimeError("geometry ordering must contain every camera once")
    return ordered_camera_ids


def load_data(args):

    K, depths = None, None
    near_clip = None
    i_sparse_unseen = np.array([], dtype=np.int64)  # 학습 제외 view 저장 배열; 결과 확인용

    if args.dataset_type == 'llff':
        images, depths, poses, bds, render_poses, i_test = load_llff_data(
                args.datadir, args.factor, args.width, args.height,
                recenter=True, bd_factor=args.bd_factor,
                spherify=args.spherify,
                load_depths=args.load_depths,
                movie_render_kwargs=args.movie_render_kwargs, args=args)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        # by seok. set llffhold as 50 to use full image set as training data
        # args.llffhold = 50
        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        # i_test = [1, 2]
        # i_test = []
        i_val = i_test

        # 정렬 후 stride 적용
        all_view_indices = np.arange(int(images.shape[0]), dtype=np.int64)
        configured_stride = getattr(args, 'train_view_stride', None)
        if configured_stride is None:
            i_train = all_view_indices
        else:
            if isinstance(configured_stride, bool) or not isinstance(
                    configured_stride, (int, np.integer)):
                raise ValueError("train_view_stride must be a positive integer")
            train_view_stride = int(configured_stride)
            if train_view_stride < 1:
                raise ValueError("train_view_stride must be at least 1")

            train_candidate_indices = all_view_indices

            geometry_ordered_ids = _build_geometry_order(   # 카메라 pose 기반 정렬
                poses[train_candidate_indices],
                train_candidate_indices,
            )

            i_train = geometry_ordered_ids[::train_view_stride].copy()  # stride 적용
            i_sparse_unseen = np.setdiff1d(     # 학습 제외 view 저장
                train_candidate_indices,
                i_train,
            )

            i_test = np.array([], dtype=np.int64)
            i_val = np.array([], dtype=np.int64)
        # i_train = np.array([i for i in np.arange(int(images.shape[0])) if
        #                 (i not in i_test and i not in i_val)])

        # #////////////////////////////////////////////////////////////////
        # for i in range(len(i_train)):
        #     if i_train[i] + 10 <= 31:
        #         i_train[i] += 10
        #     else:
        #         i_train[i] -= 22
        #////////////////////////////////////////////////////////////////


        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        else:
            near_clip = max(np.ndarray.min(bds) * .9, 0)
            _far = max(np.ndarray.max(bds) * 1., 0)
            near = 0
            far = inward_nearfar_heuristic(poses[i_train, :3, 3])[1]
            print('near_clip', near_clip)
            print('original far', _far)
        print('NEAR FAR', near, far)

        if depths == 0:
            depths = np.zeros_like(images[..., :1])

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip, args=args)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = 2., 6.

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'blendedmvs':
        images, poses, render_poses, hwf, K, i_split = load_blendedmvs_data(args.datadir)
        print('Loaded blendedmvs', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

        assert images.shape[-1] == 3

    elif args.dataset_type == 'tankstemple':
        images, poses, render_poses, hwf, K, i_split = load_tankstemple_data(
                args.datadir, movie_render_kwargs=args.movie_render_kwargs)
        print('Loaded tankstemple', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
#        i_test = [0]

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)
        near_clip = near

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'nsvf':
        images, poses, render_poses, hwf, i_split = load_nsvf_data(args.datadir)
        print('Loaded nsvf', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])
        near_clip = near

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'deepvoxels':
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.scene, basedir=args.datadir, testskip=args.testskip)
        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R - 1
        far = hemi_R + 1
        assert args.white_bkgd
        assert images.shape[-1] == 3

    elif args.dataset_type == 'co3d':
        # each image can be in different shapes and intrinsics
        images, masks, poses, render_poses, hwf, K, i_split = load_co3d_data(args)
        print('Loaded co3d', args.datadir, args.annot_path, args.sequence_name)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)

        for i in range(len(images)):
            if args.white_bkgd:
                images[i] = images[i] * masks[i][...,None] + (1.-masks[i][...,None])
            else:
                images[i] = images[i] * masks[i][...,None]

    elif args.dataset_type == 'nerfpp':
        images, poses, render_poses, hwf, K, i_split = load_nerfpp_data(args.datadir)
        print('Loaded nerf_pp', images.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near_clip, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0.02)
        near = 0

    elif args.dataset_type == 'replica':
        images, poses, render_poses, hwf, i_split = load_replica_data(args.datadir, args.half_res, args.testskip, args=args, \
                                                                                        spherify=args.spherify,movie_render_kwargs=args.movie_render_kwargs)
        print('Loaded replica', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)
        near_clip = near

        print('NEAR FAR', near, far)


        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'lerf':
        images, poses, render_poses, hwf, K, i_split = load_lerf_data(args.datadir, args.factor,movie_render_kwargs=args.movie_render_kwargs)
        print('Loaded lerf', images.shape, render_poses.shape, hwf[:2], args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)
        near_clip = near

        print('NEAR FAR', near, far)


        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks,
        near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        i_sparse_unseen=i_sparse_unseen,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,
        irregular_shape=irregular_shape
    )
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far
