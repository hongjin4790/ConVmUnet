# ConVmUnet
피부 모공 검출을 위한 ConVmUnet 영상 분할 기법 [논문](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003190838)의 코드 저장소입니다.

 # Abstract
본 논문에서는 피부 모공 검출을 위하여 피부 영상에서 영상 분할 성능을 향상시킨 ConVmUnet 모델을 제안한다. 제안 모델은 
CNN 모델과 Vmamba 모델의 VSS 블록을 융합한 Unet 구조로 인코더 부분의 CNN 모델에서 피부 영상의 특성을 추출하고 VSS
블록에서 추출된 특징들을 통합 처리 및 복원하여 피부 영상에서 모공을 구분하고 검출한다. 6,079장의 피부 영상 데이터 세트를 
사용하여 학습 및 성능 평가(IoU, Dice Score)를 수행하였고, 기존 Unet 계열의 모델들과 비교하여 향상된 성능을 보여주었다. 향
후 다양한 피부 영상 데이터를 학습하여 모공 이외의 붉음증, 피부결 등과 같은 특성을 구분하고 검출할 수 있도록 제안 모델을 확
장시킬 예정이다.

# Dataset
![image](https://github.com/user-attachments/assets/f8ed5ce8-2edf-4d77-a759-6e1f553bfb9e)

피부 데이터는 실제 소비자들이 촬영한 피부 영상 6,079장을 사용하였다. 학습 데이터 4,863장, 검증 데이터 608장, 테스트 데이터 608장으로 8:1:1 비율로 나누었다.
모든 피부 데이터의 해상도는 256x256이며 마스크를 생성하기 위해 기존 연구에서 사용한 모공 특징 추출 영상처리 기법을 사용하였다.


# Model Structure
![image](https://github.com/user-attachments/assets/4dc436f5-193e-4dd1-b237-5082353c3b88)

ConVmUnet은 CNN과 Vmamba 기반의 VSS(Visual State Space)블록을 융합한 Encoder-Decoder 구조의 딥러닝 모델이다.

- Encoder
  - 2D Convolution 레이어로 국소적 특징 추출
  - 추출된 특징은 패치 임베딩 과정을 거쳐 VSS 블록에 입력

- VSS 블록
  - 국소적 정보와 전역 정보를 동시에 처리
  - SS2D 블록을 통해 중요한 부분만 집중 분석하여 계산 효율성 향상
 
- SS2D 블록 구조
  - 스캔 확장: 입력을 상/하/좌/우 방향으로 시퀀스화하여 정보 수집
  - S6 블록: 다양한 특징을 추출하고 문맥 기반 가중치를 부여
  - 스캔 병합: 가중치 특징 맵을 병합 및 원래 크기로 복원
 
- Decoder
  - VSS 블록의 출력은 Hidden Feature 블록에서 512차원으로 변환 후 업샘플링
  - Skip connection을 통해 Encoder의 feature와 결합
  - Segmentation Head를 통해 각 픽셀의 클래스를 예측하는 분할 맵 출력

