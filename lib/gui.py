import logging
from collections import deque

# logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import imageio
import time
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
import dash
from dash import Dash, Input, Output, dcc, html, State
from dash.exceptions import PreventUpdate
from .self_prompting import grounding_dino_prompt
from model_assets import load_yolo_model

# by seok
from torchvision.utils import save_image
from lib import sam3d
from . import utils
import PIL
import torchvision.transforms as transforms

# by young
from scipy.spatial.transform import Rotation as R

# model_yolo = YOLO('yolov8n.pt')
model_yolo = load_yolo_model()
best_view_idx = None  # 전역 변수로 선언

def compute_pose_distance(pose1, pose2, w=0.5):
    pose1_cpu = pose1.cpu().numpy() if isinstance(pose1, torch.Tensor) else pose1
    pose2_cpu = pose2.cpu().numpy() if isinstance(pose2, torch.Tensor) else pose2
    r1, r2 = R.from_matrix(pose1_cpu[:3, :3]), R.from_matrix(pose2_cpu[:3, :3])
    angle_diff = (r1.inv() * r2).magnitude()
    trans_diff = np.linalg.norm(pose1_cpu[:3, 3] - pose2_cpu[:3, 3])
    
    return trans_diff + angle_diff

def mark_image(_img, points):
    assert(len(points) > 0)
    img = _img.copy()
    r = 10
    mark_color = np.array([255, 0, 0]).reshape(1, 1, 3)
    for i in range(len(points)):
        point = points[i]
        img[point[1]-r:point[1]+r+1, point[0]-r:point[0]+r+1] = mark_color
    return img

def draw_figure(fig, title, animation_frame=None):
    fig = px.imshow(fig, animation_frame=animation_frame)
    if animation_frame is not None:
        # fig.update_layout(sliders = [{'visible': False}])
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 33
    fig.update_layout(title_text=title, showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

def find_view_with_max_objects(data_dict, yolo_model, target_class_id):
    """
    모든 객체가 가장 잘 보이는 뷰를 선택하는 함수
    Args:
        data_dict: 학습 데이터 (images, poses, etc.)
        yolo_model: YOLO 모델 객체
        target_class_id: 타겟 클래스 ID
    Returns:
        best_view_idx: 가장 많은 객체가 감지된 뷰의 로컬 인덱스
        max_score: 해당 뷰의 종합 점수
    """
    # max_instances = 0
    # best_view_idx = 0

    # for idx in range(len(data_dict['i_train'])):
    #     img = data_dict['images'][idx, :, :, :].numpy()
    #     img = utils.to8b(img)
    #     h, w, c = img.shape

    #     # YOLO로 객체 감지
    #     results = yolo_model.predict(source=img, imgsz=(h, w))
    #     num_instances = len(results[0].boxes)  # 감지된 객체 수

    #     if num_instances > max_instances:
    #         max_instances = num_instances
    #         best_view_idx = idx

    # return best_view_idx, max_instances

    max_score = -float('inf')
    best_view_idx = 0
    train_ids = np.asarray(data_dict['i_train'], dtype=np.int64)

    for local_idx, camera_id in enumerate(train_ids):
        img = data_dict['images'][camera_id].numpy()
        img = utils.to8b(img)
        h, w, c = img.shape
        
        # YOLO 추론 (이미지 크기 자동 조정 방지)
        results = yolo_model.predict(
            source=img, 
            imgsz=(h, w),  # 원본 해상도 유지
            classes=target_class_id,  # 특정 클래스만 검출
            stream=False  # 단일 이미지 처리
        )
        
        # 신뢰도 및 객체 수 계산
        if len(results[0].boxes) == 0:
            continue  # 객체 미검출 시 건너뜀
            
        confidences = results[0].boxes.conf.cpu().numpy()
        num_instances = len(results[0].boxes)
        
        # 종합 점수 계산 (신뢰도 70% + 객체 수 30%)
        conf_sum = np.sum(confidences)
        score = (conf_sum / len(confidences) * 0.7) + (num_instances * 0.3) # 평균 confidence로 수정

        # 최대 점수 갱신
        if score > max_score:
            max_score = score
            best_view_idx = local_idx

    return best_view_idx, max_score


class Sam3dGUI:
    def __init__(self, Seg3d, debug=False):
        ctx = {
            'num_clicks': 0, 
            'click': [], 
            'cur_img': None, 
            'btn_clear': 0, 
            'btn_text': 0, 
            'prompt_type': 'point',
            'show_rgb': False
            }
        self.ctx = ctx
        self.Seg3d = Seg3d
        self.debug = debug

        self.train_idx = 0

    def run(self):
        global best_view_idx  # 전역 변수 사용
        init_rgb = self.Seg3d.init_model()

        # 모든 객체가 잘 보이는 뷰를 선택
        best_view_idx, max_score = find_view_with_max_objects(self.Seg3d.data_dict, model_yolo, self.Seg3d.target_class_id)
        best_view_camera_id = int(self.Seg3d.data_dict['i_train'][best_view_idx])

        logging.info(f"Best view with max objects: {best_view_idx} (Camera: {best_view_camera_id}, Score: {max_score})")

        # 선택된 뷰에서 초기 이미지 설정
        init_rgb = self.Seg3d.data_dict['images'][best_view_camera_id, :, :, :].numpy()
        init_rgb = utils.to8b(init_rgb)
        self.Seg3d.predictor.set_image(init_rgb)

        self.ctx['cur_img'] = init_rgb
        self.run_app(sam_pred=self.Seg3d.predictor, ctx=self.ctx, init_rgb=init_rgb)


    def run_app(self, sam_pred, ctx, init_rgb):
        '''
        run dash app
        '''
        def query(points=None, text=None):
            with torch.no_grad():
                if text is None:
                    input_point = points
                    input_label = np.ones(len(input_point))
                    masks, scores, logits = sam_pred.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )
                elif points is None:             
                    h, w, c = ctx['cur_img'].shape
                    results = model_yolo.predict(source=ctx['cur_img'], imgsz=(h,w), classes=self.Seg3d.target_class_id)
                    h2, w2 = results[0].masks.data[0].shape
                    m = torch.zeros([h, w])
                    for j, img in enumerate(results[0].masks.data):
                        # save_image(img, f'result_{j}.png')
                        m += img[(h2-h)//2:(h2+h)//2,:]

                    # first person only
                    # p1 0, p2 2, p3 5, p4 1, p5 4, p6 3
                    idx_select = 0
                    m = torch.zeros([h, w])
                    img = results[0].masks.data[idx_select]
                    m = img[(h2-h)//2:(h2+h)//2,:]

                    masks = torch.zeros([c, h, w]).cpu().numpy()
                    masks[0:3,:,:] = m.type(torch.bool).cpu().numpy()

                    # save_image(m, 'yolo_00.png')

                else:
                    raise NotImplementedError

            fig1 = (255*masks[0, :, :, None]*0.6 + ctx['cur_img']*0.4).astype(np.uint8)
            fig2 = (255*masks[1, :, :, None]*0.6 + ctx['cur_img']*0.4).astype(np.uint8)
            fig3 = (255*masks[2, :, :, None]*0.6 + ctx['cur_img']*0.4).astype(np.uint8)
            fig1 = draw_figure(fig1, 'mask0')
            fig2 = draw_figure(fig2, 'mask1')
            fig3 = draw_figure(fig3, 'mask2')

            if text is None:
                fig0 = mark_image(ctx['cur_img'], points)
            else:
                fig0 = ctx['cur_img']
            fig0 = draw_figure(fig0, 'original_image')

            return  masks, fig0, fig1, fig2, fig3
        
        # _, fig0, fig1, fig2, fig3, desc = query(np.array([[100, 100], [101, 101]]))
        self.ctx['fig0'] = draw_figure(init_rgb, 'original_image')
        self.ctx['fig1'] = draw_figure(np.zeros_like(init_rgb), 'mask0')
        self.ctx['fig2'] = draw_figure(np.zeros_like(init_rgb), 'mask1')
        self.ctx['fig3'] = draw_figure(np.zeros_like(init_rgb), 'mask2')
        self.ctx['fig_seg_rgb'] = draw_figure(np.zeros_like(init_rgb), 'Masked image in Training')
        self.ctx['fig_sam_mask'] = draw_figure(np.zeros_like(init_rgb), 'SAM Mask with Prompts in Training')
        self.ctx['fig_masked_rgb'] = draw_figure(np.zeros_like(init_rgb), 'Masked RGB')
        self.ctx['fig_seged_rgb'] = draw_figure(np.zeros_like(init_rgb), 'Seged RGB')
        
        app = dash.Dash(
            __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
        )
        app.layout = html.Div(
            style={"height": "100%"},
            children=[
            html.Div(className="container", children=[
                html.Div(className="row", children=[
                    html.Div(className="two columns",style={"padding-bottom": "5%"},children=[
                        html.Div([html.H3(['SAM Init'])]),
                        html.Br(),

                        html.H5('Prompt Type:'),
                        html.Div([
                            dcc.Dropdown(
                                id = 'prompt_type',
                                options = [{'label': 'Points', 'value': 'point'}, 
                                        {'label': 'Text', 'value': 'text'},],
                                value = 'point'),
                                html.Div(id = 'output-prompt_type')
                        ]),
                        html.Br(),

                        html.H5('Point Prompts:'),
                        html.Button('Clear Points', id='btn-nclicks-clear', n_clicks=0),
                        html.Br(),

                        html.H5('Text Prompt:'),
                        html.Div([
                            dcc.Input(id='input-text-state', type='text', value='none'),
                            html.Button(id='submit-button-state', n_clicks=0, children='Generate'),
                            html.Div(id='output-state-text')
                        ]),
                        html.Br(),

                        html.H5('Please select the mask:'),
                        html.Div([
                            dcc.RadioItems(['mask0', 'mask1', 'mask2'], id='sel_mask_id', value=None)
                        ], style={'display': 'flex'}),
                        html.Br(),

                        html.H5(id='container-sel-mask'),
                    ]),
                    html.Div(className="ten columns",children=[
                        html.Div(children=[
                            dcc.Graph(id='main_image', figure=self.ctx['fig0'])
                        ], style={'display': 'inline-block', 'width': '40%'}),

                        html.Div(children=[
                            dcc.Graph(id='mask0', figure=self.ctx['fig1'])
                        ], style={'display': 'inline-block', 'width': '40%'}),

                        html.Div(children=[
                            dcc.Graph(id='mask1', figure=self.ctx['fig2'])
                        ], style={'display': 'inline-block', 'width': '40%'}),

                        html.Div(children=[
                            dcc.Graph(id='mask2', figure=self.ctx['fig3'])
                        ], style={'display': 'inline-block', 'width': '40%'}),
                    ])
                ])
            ]),

            html.Div(className="container", children=[
                html.Div(className="row", children=[
                    html.Div(className="two columns",style={"padding-bottom": "5%"},children=[
                        html.Div([html.H3(['SA3D Training'])]),
                        html.Br(),

                        html.Button('Start Training', id='btn-nclicks-training', n_clicks=0),
                        html.Div(id='container-button-training', style={'display': 'inline-block'}),
                        ]),

                    html.Div(className="ten columns",children=[
                        html.Div(children=[
                            dcc.Graph(id='seg_rgb', figure=self.ctx['fig_seg_rgb'])
                        ], style={'display': 'inline-block', 'width': '40%'}),

                        html.Div(children=[
                            dcc.Graph(id='sam_mask', figure=self.ctx['fig_sam_mask'])
                        ], style={'display': 'inline-block', 'width': '40%'}),
                    ]),

                    dcc.Interval(
                        id='interval-component',
                        interval=1*1000,  # in milliseconds
                        n_intervals=0),
                ])
            ]),

            html.Div(className="container", children=[
                html.Div(className="row", children=[
                    html.Div(className="two columns",style={"padding-bottom": "5%"},children=[
                        html.Div([html.H3(['SA3D Rendering Results'])]),
                        html.Br(),
                        ]),

                    html.Div(className="ten columns",children=[
                        html.Div(children=[
                            dcc.Graph(id='masked_rgb', figure=self.ctx['fig_masked_rgb'])
                        ], style={'display': 'inline-block', 'width': '40%'}),

                        html.Div(children=[
                            dcc.Graph(id='seged_rgb', figure=self.ctx['fig_seged_rgb'])
                        ], style={'display': 'inline-block', 'width': '40%'}),
                    ]),
                ])
            ])
            
        ])

        @app.callback(Output('output-prompt_type', 'children'), [Input('prompt_type', 'value')])
        def update_prompt_type(value):
            self.ctx['prompt_type'] = value
            if value != 'point':
                ctx['click'] = []
                ctx['num_clicks'] = 0
            return f"Type {value} is chosen"
        

        @app.callback(
            Output('main_image', 'figure'),
            Output('mask0', 'figure'),
            Output('mask1', 'figure'),
            Output('mask2', 'figure'),
            Output('output-state-text', 'children'),
            Input('main_image', 'clickData'),
            Input('btn-nclicks-clear', 'n_clicks'),
            Input('submit-button-state', 'n_clicks'),
            State('input-text-state', 'value')
        )
        def update_prompt(clickData, btn_point, btn_text, text):
            '''
            update mask
            '''
            if self.ctx['prompt_type'] == 'point':
                if clickData is None and btn_point == self.ctx['btn_clear']:
                    raise PreventUpdate

                if btn_point > self.ctx['btn_clear']:
                    self.ctx['btn_clear'] += 1
                    ctx['click'] = []
                    ctx['num_clicks'] = 0
                    return self.ctx['fig0'], self.ctx['fig1'], self.ctx['fig2'], self.ctx['fig3'], 'none'
                
                ctx['num_clicks'] += 1
                ctx['click'].append(np.array([clickData['points'][0]['x'], clickData['points'][0]['y']]))
                
                ctx['saved_click'] = np.stack(ctx['click'])
                masks, fig0, fig1, fig2, fig3 = query(ctx['saved_click'])
                ctx['masks'] = masks
                return fig0, fig1, fig2, fig3, 'none'
            
            elif self.ctx['prompt_type'] == 'text':
                if btn_text > self.ctx['btn_text']:
                    self.ctx['btn_text'] += 1
                    self.ctx['text'] = text
                    masks, fig0, fig1, fig2, fig3 = query(points=None, text=text)
                    ctx['masks'] = masks
                    return fig0, fig1, fig2, fig3, u'''
                        Input text is "{}"
                    '''.format(text)
                else:
                    raise PreventUpdate
            else:
                raise NotImplementedError

        @app.callback(
            Output("container-sel-mask", 'children'),
            Input("sel_mask_id", 'value')
        )
        def update_graph(radio_items):
            if radio_items == 'mask0':
                ctx['select_mask_id'] = 0
                return html.Div("you select mask0")
            elif radio_items == 'mask1':
                ctx['select_mask_id'] = 1
                return html.Div("you select mask1")
            elif radio_items == 'mask2':
                ctx['select_mask_id'] = 2
                return html.Div("you select mask2")
            else:
                raise PreventUpdate
            
        @app.callback(
            Output('seg_rgb', 'figure'),
            Output('sam_mask', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def displaySeg(n):
            if self.ctx['show_rgb']:
                self.ctx['show_rgb'] = False
                fig_seg_rgb = draw_figure(self.ctx['fig_seg_rgb'], 'Masked image in Training')
                fig_sam_mask = draw_figure(self.ctx['fig_sam_mask'], 'SAM Mask with Prompts in Training')
                return fig_seg_rgb, fig_sam_mask
            else:
                raise PreventUpdate
        

        @app.callback(
            Output('container-button-training', 'children'),
            Output('masked_rgb', 'figure'),
            Output('seged_rgb', 'figure'),
            Input('btn-nclicks-training', 'n_clicks')
        )
        def start_training(btn):
            if btn < 1:
                return html.Div("Press to start training"), self.ctx['fig_masked_rgb'], self.ctx['fig_seged_rgb']
            else:
                logging.debug("Starting training process")
                # optim in the first view

                original_train_ids = np.asarray(
                    self.Seg3d.data_dict['i_train'],
                    dtype=np.int64,
                ).copy()
                all_view_indices = list(range(len(original_train_ids)))
                print("Available view indices:", all_view_indices)
                print("Best view index:", best_view_idx)

                # # 1. 실제 뷰 ID 리스트
                # all_view_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

                # print("valid_views:", all_view_ids)  # valid_views 리스트 출력
                # print("best_view_idx:", best_view_idx)  # best_view_idx 값 출력

                # 포즈 거리 기반 정렬 적용
                sorted_indices = [best_view_idx]
                available_indices = list(range(len(all_view_indices)))
                available_indices.remove(best_view_idx)

                # 히스토리 윈도우 초기화
                history_window = deque(maxlen=4)
                history_window.append(best_view_idx)

                # 포즈 거리 기반 정렬
                poses = self.Seg3d.data_dict['poses'][original_train_ids]

                while available_indices:
                    # 가중치 계산
                    weighted_distances = {}
                    for candidate in available_indices:
                        total = 0.0
                        weight_sum = 0.0
                        for k, prev_idx in enumerate(reversed(history_window)):
                            weight = 0.5 ** (k+1)
                            distance = compute_pose_distance(poses[prev_idx], poses[candidate])
                            total += weight * distance
                            weight_sum += weight
                        weighted_distances[candidate] = total / weight_sum  # 정규화
                    # 최소 가중 거리 선택
                    next_idx = min(weighted_distances.items(), key=lambda x: x[1])[0]
                    # 히스토리 업데이트
                    sorted_indices.append(next_idx)
                    available_indices.remove(next_idx)
                    history_window.append(next_idx)

                # 첫 번째 학습을 위한 정렬된 인덱스 저장
                first_order_local = np.asarray(sorted_indices, dtype=np.int64)
                i_train_sorted_1 = original_train_ids[first_order_local].tolist()
                self.Seg3d.data_dict['i_train'] = i_train_sorted_1

                # start_idx = best_view_idx
                
                # # 3. best_view_idx부터 끝까지 순차적으로 진행
                # view_order = all_view_ids[start_idx:] + all_view_ids[:start_idx]

                # print("view_order:", view_order)  # best_view_idx 값 출력

                # view_id_to_index = {view_id: index for index, view_id in enumerate(all_view_ids)}
                # index_to_view_id = {index: view_id for index, view_id in enumerate(all_view_ids)}
                
                # i_train_sorted_1 = [view_id_to_index[view_id] for view_id in view_order]

                # self.Seg3d.data_dict['i_train'] = i_train_sorted_1  # 학습 순서 갱신

                # render_poses, HW, Ks 갱신
                self.Seg3d.update_render_poses()

                self.train_idx = 0
                # `train_step` 호출
                self.Seg3d.train_step(self.train_idx, sam_mask=ctx['masks'][ctx['select_mask_id']])
                self.train_idx += 1  # train_idx 증가

                while True:
                    rgb, sam_prompt, is_finished = self.Seg3d.train_step(self.train_idx)
                    self.train_idx += 1
                    self.ctx['fig_seg_rgb'] = rgb
                    if sam_prompt is not None: # sam_prompt가 None이 아닐 때만 업데이트
                        self.ctx['fig_sam_mask'] = sam_prompt
                    self.ctx['show_rgb'] = True
                    logging.info(f"Updated fig_seg_rgb and fig_sam_mask at train_idx {self.train_idx}")
                    
                    if is_finished:
                        break

                # 학습순서선정/학습수행 플래그 설정
                self.Seg3d.vsgflag = True  # 할당 연산자

                # 첫 번째 학습에서 사용한 인덱스 저장
                first_train_indices = i_train_sorted_1.copy()
                
                # by seok: sort view list according to the matching score
                matching_score = self.Seg3d.matching_score
                if not matching_score:
                    raise RuntimeError("No valid matching scores were recorded during the first pass.")

                best_matching_camera_id = max(matching_score, key=matching_score.get)

                camera_to_first_local = {
                    int(camera_id): local_idx
                    for local_idx, camera_id in enumerate(first_train_indices)
                }
                best_matching_local_idx = camera_to_first_local[best_matching_camera_id]

                sorted_indices = [best_matching_local_idx]
                available_indices = list(range(len(first_train_indices)))
                available_indices.remove(best_matching_local_idx)

                history_window = deque(maxlen=4)  # 최근 4개 뷰 추적
                history_window.append(best_matching_local_idx)

                # by young : sort view list according to the pose distances
                # 포즈 거리 기반으로 정렬된 순서에 따라 학습 진행
                poses = self.Seg3d.data_dict['poses'][self.Seg3d.data_dict['i_train']]

                while available_indices:
                    # 가중치 계산
                    weighted_distances = {}
                    for candidate in available_indices:
                        total = 0.0
                        weight_sum = 0.0
                        for k, prev_idx in enumerate(reversed(history_window)):
                            weight = 0.5 ** (k+1)
                            distance = compute_pose_distance(poses[prev_idx], poses[candidate])
                            total += weight * distance
                            weight_sum += weight
                        weighted_distances[candidate] = total / weight_sum  # 정규화
                    # 최소 가중 거리 선택
                    next_idx = min(weighted_distances.items(), key=lambda x: x[1])[0]
                    # 히스토리 업데이트
                    sorted_indices.append(next_idx)
                    available_indices.remove(next_idx)
                    history_window.append(next_idx)

                # 두 번째 학습을 위한 정렬된 인덱스
                second_order_local = np.asarray(sorted_indices, dtype=np.int64)
                i_train_sorted_2 = np.asarray(first_train_indices)[second_order_local].tolist()

                self.Seg3d.data_dict['i_train'] = i_train_sorted_2

                print(f"Selected Instance Matching Scores: {self.Seg3d.matching_score}")
                print(f"Sorted Matching Score Order: {[matching_score.get(camera_id) for camera_id in i_train_sorted_2]}")
                print(f"Initial view (highest matching score): {best_matching_camera_id}")
                print(f"New training order: {i_train_sorted_2}")
                
                # render_poses, HW, Ks 갱신
                self.Seg3d.update_render_poses()

                second_seed_camera_id = int(i_train_sorted_2[0])
                if second_seed_camera_id != best_matching_camera_id:
                    raise RuntimeError("The second-pass order does not start from the best camera.")

                img = self.Seg3d.data_dict['images'][second_seed_camera_id].numpy()
                img = utils.to8b(img)
                h, w, c = img.shape
                results = model_yolo.predict(
                    source=img,
                    imgsz=(h, w),
                    classes=self.Seg3d.target_class_id,
                )
                if (
                    not results
                    or not results[0].boxes
                    or len(results[0].boxes) == 0
                    or results[0].masks is None
                    or len(results[0].masks.data) == 0
                ):
                    raise RuntimeError("No objects were detected in the second-pass seed view.")

                idx_select = self.Seg3d.selected_instance_by_camera.get(
                    second_seed_camera_id
                )
                if idx_select is None or idx_select >= len(results[0].masks.data):
                    raise RuntimeError("The selected instance is unavailable in the second-pass seed view.")

                h2, w2 = results[0].masks.data[idx_select].shape
                mask_img = results[0].masks.data[idx_select]
                m = mask_img[(h2 - h) // 2:(h2 + h) // 2, :]

                # 마스크 이미지 저장
                # save_image(m, f'yolo_0.png')
                masks = torch.zeros([c, h, w]).cpu().numpy()
                masks[0:3, :, :] = m.type(torch.bool).cpu().numpy()

                # train_idx 초기화
                self.train_idx = 0
                # 학습 진행
                self.Seg3d.train_step(self.train_idx, sam_mask=masks)
                self.train_idx += 1

                # cross-view training
                while True:
                    rgb, sam_prompt, is_finished = self.Seg3d.train_step(self.train_idx)
                    self.train_idx += 1
                    self.ctx['fig_seg_rgb'] = rgb
                    self.ctx['fig_sam_mask'] = sam_prompt
                    self.ctx['show_rgb'] = True
                    logging.info(f"Updated fig_seg_rgb and fig_sam_mask at train_idx {self.train_idx}")
                    if is_finished:
                        break

                # 학습 체크포인트 저장
                self.Seg3d.save_ckpt()

                # by seok get rendered mask
                # 결과 마스크 계산 및 IoU 측정
                # f = open("iou.txt", "w")
                # avg_IoU = 0
                # render_poses, HW, Ks = sam3d.fetch_seg_poses(self.Seg3d.args.seg_poses, self.Seg3d.data_dict)
                # for idx in range(len(render_poses)):
                #     rgb, depth, bgmap, seg_m, dual_seg_m = self.Seg3d.render_view(idx, [render_poses, HW, Ks])
                #     tmp_rendered_mask = seg_m.detach().cpu().clone()
                #     tmp_rendered_mask[tmp_rendered_mask < 0] = 0
                #     tmp_rendered_mask[tmp_rendered_mask != 0] = 1
                #     # imageio.imwrite(f"mask_rendered_{idx:02d}.png", tmp_rendered_mask)

                #     # Ground Truth 마스크 로드 및 IoU 계산
                #     tf = transforms.ToTensor()
                #     current_view_id = sorted_view_ids[idx]
                #     m = tf(PIL.Image.open(f'../Labeling/Set11/masks/{current_view_id:02d}_p5.png'))  # 수정
                #     m_numpy = m.cpu().numpy()
                #     if m_numpy.ndim == 3 and m_numpy.shape[0] == 1:  # 단일 채널 텐서
                #         m_numpy = m_numpy.squeeze(axis=0)  # 2D 배열로 변환
                #     elif m_numpy.ndim == 2:
                #         pass  # 이미 2D 배열
                #     else:
                #         raise ValueError("Input tensor has unsupported dimensions for saving.")
                #     imageio.imwrite(f"GT_sorted_{idx:02d}.png", m_numpy)
                #     tmp_IoU = utils.cal_IoU(m.cpu(), tmp_rendered_mask.squeeze())
                #     avg_IoU += tmp_IoU
                #     f.write(f"{tmp_IoU}\n")
                # avg_IoU /= len(render_poses)
                # f.write(f"{avg_IoU}\n")
                # f.close()

                # 학습 결과를 반환
                masked_rgb, seged_rgb = self.Seg3d.render_test()
                fig_masked_rgb = draw_figure(masked_rgb, 'Masked RGB', animation_frame=0)
                fig_seged_rgb = draw_figure(seged_rgb, 'Seged RGB', animation_frame=0)

                return html.Div("Train Stage Finished! Press Ctrl+C to Exit!"), fig_masked_rgb, fig_seged_rgb
            
        
        #app.run_server(debug=self.debug)
        app.run_server(debug=False, dev_tools_ui=False, dev_tools_props_check=False)

if __name__ == '__main__':
    from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
    class Sam_predictor():
        def __init__(self, device):
            sam_checkpoint = "./dependencies/sam_ckpt/sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
            self.predictor = SamPredictor(self.sam)
            print('sam inited!')
            # pass

        def forward(self, points, multimask_output=True, return_logits=False):
            # self.predictor.set_image(image)
            # input_point = np.array([[x, y], [x + 1, y + 1]]) # TODO, add interactive mode
            input_point = points
            input_label = np.ones(len(input_point))

            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=multimask_output,
                return_logits=return_logits
            )
            return masks
        
    image = cv2.cvtColor(cv2.imread('data/nerf_llff_data/fern/images_4/image000.png'), cv2.COLOR_BGR2RGB)
    sam_pred = Sam_predictor(torch.device('cuda'))
    sam_pred.predictor.set_image(image)
    video = np.stack(imageio.mimread('logs/llff/fern/render_train_coarse_segmentation_gui/video.rgbseg_gui.mp4'))
    gui = Sam3dGUI(None, debug=True)
    gui.ctx['cur_img'] = image
    gui.ctx['video'] = video
    gui.run_app(sam_pred.predictor, gui.ctx, image)
