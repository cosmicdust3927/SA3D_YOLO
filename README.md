# SA3D-YOLO: 3D Segmentation with Neural Radiance Fields

SA3D (Segment Anything in 3D)와 YOLO를 결합한 3D 세그멘테이션 도구입니다. NeRF (Neural Radiance Fields)를 기반으로 2D 이미지에서 3D 객체를 자동으로 감지하고 세그멘테이션할 수 있습니다.

## 🚀 주요 기능

- **3D 신경 렌더링**: NeRF를 사용한 고품질 3D 재구성
- **자동 객체 감지**: YOLO를 통한 객체 자동 감지
- **3D 세그멘테이션**: SAM (Segment Anything Model) 기반 3D 세그멘테이션
- **인터랙티브 GUI**: 웹 기반 사용자 인터페이스
- **다양한 데이터셋 지원**: LLFF, Blender, Tank & Temple, DeepVoxels, Replica, LERF 등

## 📋 요구사항

- Python 3.10
- CUDA 11.6+
- GPU with 8GB+ VRAM (권장)

## 🛠️ 설치

### 1. 저장소 클론
```bash
git clone https://github.com/cosmicdust3927/SA3D_YOLO.git
cd SA3D_YOLO
```

### 2. 환경 설정
```bash
conda create -n sa3d python=3.10
conda activate sa3d
```

### 3. CUDA 및 PyTorch 설치
```bash
conda install -c anaconda -c conda-forge cudatoolkit==11.6
conda install -c anaconda cudnn
conda install -c conda-forge cudatoolkit-dev
pip install torch==1.12.1+cu116 \
    torchvision==0.13.1+cu116 \
    torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116
```

### 4. PyTorch 확장 및 Python 패키지 설치
```bash
pip install torch_scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
pip install -r requirements.txt
```

### 5. SAM 설치
```bash
mkdir -p dependencies/sam_ckpt
cd dependencies/sam_ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
cd ../..
```

### 6. Grounding-DINO 설치
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
mkdir -p weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../../..
```

## 📁 데이터 준비

### 데이터 구조
```
SA3D_YOLO/
├── data/
│   └── [your_dataset]/
│       ├── images/
│       ├── poses_bounds.npy
│       └── ...
├── yolov8x-seg.pt  # YOLO 모델 (자동 다운로드)
└── dependencies/sam_ckpt/sam_vit_h_4b8939.pth
```

### YOLO 모델 다운로드
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt
```

## 🎯 사용법

### 1. NeRF 모델 학습
```bash
python run.py --config=configs/nerf_unbounded/Set1.py --stop_at=20000 --render_video --i_weights=10000
```

### 2. 3D 세그멘테이션 실행 (GUI)
```bash
python run_seg_gui.py --config=configs/nerf_unbounded/seg_Set1.py --segment --sp_name=_gui --num_prompts=20 --render_opt=train --save_ckpt
```

### 3. 결과 렌더링
```bash
python run_seg_gui.py --config=configs/nerf_unbounded/seg_Set1.py --segment --sp_name=_gui --num_prompts=20 --render_only --render_opt=video --dump_images --seg_type seg_img seg_density
```

## 🔧 설정 파일

- `configs/default.py`: 기본 설정
- `configs/seg_default.py`: 세그멘테이션 기본 설정
- `configs/nerf_unbounded/`: 데이터셋별 설정

## 📊 지원 데이터셋

- **LLFF**: Local Light Field Fusion 데이터셋
- **Blender**: Blender 합성 데이터셋
- **Tank & Temple**: Tank & Temple 데이터셋
- **DeepVoxels**: DeepVoxels 데이터셋
- **Replica**: Replica 데이터셋
- **LERF**: Language Embedded Radiance Fields 데이터셋

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🙏 감사의 말

- [SA3D](https://github.com/Jumpat/SegmentAnythingin3D): 원본 SA3D 구현
- [Segment Anything](https://github.com/facebookresearch/segment-anything): SAM 모델
- [YOLOv8](https://github.com/ultralytics/ultralytics): YOLO 객체 감지
- [NeRF](https://github.com/bmild/nerf): Neural Radiance Fields

## 📞 문의

문제가 있거나 질문이 있으시면 [Issues](https://github.com/cosmicdust3927/SA3D_YOLO/issues) 페이지에 문의해주세요.
