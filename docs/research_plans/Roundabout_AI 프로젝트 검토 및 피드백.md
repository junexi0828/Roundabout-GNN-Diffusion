# **SDD Death Circle 데이터셋 기반 이기종 에이전트 상호작용 예측을 위한 ST-HGNN 아키텍처 및 안전성 검증 프레임워크에 대한 심층 평가 보고서**

## **1\. 서론: 혼합 자율주행 환경에서의 예측 불확실성과 이기종성의 중요성**

자율주행 시스템과 로봇 공학의 패러다임은 단순한 객체 인식 및 회피를 넘어, 복잡한 사회적 상호작용(Social Interaction)을 이해하고 예측하는 '사회적 탐색(Social Navigation)'의 단계로 진입하고 있다. 특히, 차량, 자전거, 보행자가 혼재된 이기종(Heterogeneous) 교통 환경은 각 에이전트의 운동학적 특성과 거동 패턴이 상이하여 단일 모델로 예측하기 어려운 고난도의 문제를 제기한다. 이러한 맥락에서 제안된 'SDD(Stanford Drone Dataset) Death Circle 데이터셋을 활용한 이기종 에이전트 상호작용 예측 프로젝트'는 현대 자율주행 연구의 핵심 난제인 비정형 도로 환경에서의 다중 에이전트 궤적 예측(Multi-Agent Trajectory Prediction)을 정면으로 다루고 있다는 점에서 높은 학술적 시의성을 갖는다.1

스탠포드 대학교의 Death Circle은 원형 교차로(Roundabout)라는 특수한 토폴로지를 가지고 있어, 신호등에 의한 이산적 제어보다 에이전트 간의 연속적인 양보와 진입(Gap Acceptance) 협상이 빈번하게 발생하는 구역이다. 이곳에서는 차량의 강력한 운동 에너지와 보행자의 급격한 방향 전환성이 공존하며, 이를 정확히 모델링하기 위해서는 단순한 위치 좌표의 시계열 예측을 넘어 에이전트의 유형(Type)에 따른 상호작용의 가중치를 차별화하는 접근이 필수적이다.3

본 보고서는 제안된 **시공간 이기종 그래프 신경망(ST-HGNN, Spatio-Temporal Heterogeneous Graph Neural Network)** 구조와 **호모그래피(Homography) 변환**, 그리고 **안전 지표(Safety Metrics) 검증**이라는 세 가지 핵심 축을 2024-2025년 최신 연구 트렌드와 비교 분석한다. 특히, 단순히 예측 정확도(ADE/FDE)만을 추구하던 기존 경향에서 벗어나, 물리적 타당성과 충돌 위험도를 평가하는 안전성 검증 레이어의 도입은 본 프로젝트가 단순한 학부 수준의 구현을 넘어 석박사급의 심도 있는 연구로 발전할 잠재력을 보여준다. 본 분석은 현재의 기술적 타당성을 검토하고, 향후 구현 단계에서 반드시 고려해야 할 비교 모델(Baselines), 시각화 전략, 그리고 최신 생성형 모델(Generative Models)과의 통합 가능성을 구체적으로 제안함으로써 프로젝트의 완성도를 극대화하는 데 기여하고자 한다.

## ---

**2\. 아키텍처 평가: ST-HGNN의 타당성 및 최신 연구 동향과의 부합성**

### **2.1 동질성에서 이기종성으로의 전환 (Shift to Heterogeneity)**

과거 Social-LSTM이나 Social-GAN과 같은 1세대 딥러닝 예측 모델들은 모든 에이전트를 '움직이는 점'으로 간주하여 동일한 LSTM 가중치를 공유하는 동질적(Homogeneous) 접근 방식을 취했다. 그러나 Death Circle과 같은 환경에서는 버스의 주행 의도와 스케이트보더의 주행 의도가 본질적으로 다르다. 버스는 도로의 곡률을 따라야 하는 구속조건(Holonomic constraints)이 강한 반면, 스케이트보더는 자유도가 높다.4

제안된 **ST-HGNN**은 이러한 이기종성을 그래프 구조 내에 명시적으로 반영한다는 점에서 매우 적절한 선택이다. 이기종 그래프(Heterogeneous Graph)는 노드 타입($\\phi(v)$)과 엣지 타입($\\psi(e)$)을 구분하여, '차량-차량' 간의 메시지 패싱과 '차량-보행자' 간의 메시지 패싱에 서로 다른 가중치 행렬 $W\_{\\phi}$를 학습시킨다. 이는 최근 연구인 Trajectron++나 HeteroGNN 등에서 입증된 바와 같이, 데이터 불균형이 심한 혼합 교통 상황에서 예측 정확도를 획기적으로 향상시키는 필수적인 구조적 요소이다.6 특히 SDD 데이터셋은 차량보다 보행자와 자전거의 비율이 압도적으로 높으므로, 동질적 모델을 사용할 경우 차량의 거동 특성이 보행자 데이터에 의해 희석될 위험이 크다. ST-HGNN은 이러한 문제를 구조적으로 방지한다.8

### **2.2 그래프 신경망(GNN)과 트랜스포머(Transformer)의 융합 트렌드**

현재 궤적 예측 분야의 SOTA(State-of-the-Art)는 순수한 GNN에서 벗어나, 시계열적 의존성을 포착하는 데 탁월한 트랜스포머(Transformer) 구조를 통합하는 방향으로 나아가고 있다. AgentFormer 9와 같은 모델은 에이전트 간의 관계를 그래프가 아닌 어텐션(Attention) 메커니즘으로 평탄화(Flatten)하여 처리하지만, 이기종 에이전트 간의 관계를 명확히 구조화하기에는 GNN이 여전히 직관적인 강점을 가진다.

제안된 ST-HGNN 구조는 '공간적 관계'를 GNN으로, '시간적 흐름'을 RNN(LSTM/GRU) 또는 Temporal CNN으로 처리하는 전형적인 접근법으로 보인다. 그러나 최신 트렌드에 부합하기 위해서는 단순한 GNN이 아닌 **이종 그래프 어텐션(Heterogeneous Graph Attention)** 메커니즘이 필수적이다. 이는 단순한 거리 기반 연결이 아니라, 에이전트의 상태(속도, 방향)에 따라 상호작용의 중요도를 동적으로 조절할 수 있어야 함을 의미한다. 예를 들어, Death Circle 진입로에서 차량은 자신의 진행 방향에 있는 자전거에는 높은 어텐션을 주어야 하지만, 등 뒤에 있는 보행자에게는 낮은 어텐션을 주어야 한다. 이러한 '비등방성(Anisotropic)' 상호작용 모델링이 ST-HGNN 내에 구현되어야 한다.10

### **2.3 정적 환경 정보(Scene Context)의 통합 부재에 대한 우려**

현재 제안서에서 가장 보완이 필요한 부분은 '맵(Map)' 정보의 통합이다. Death Circle은 원형 교차로라는 강력한 기하학적 제약 조건을 가진다. 차량은 중앙 구조물을 가로질러 갈 수 없으며, 차선을 따라 이동해야 한다. 최근 연구인 GraphAD 12나 LaneGCN 등은 에이전트 간의 상호작용뿐만 아니라, **에이전트와 차선(Lane) 간의 상호작용**을 이기종 그래프의 일부로 모델링한다.

제안된 ST-HGNN에 'Lane Node' 또는 'Scene Context Node'를 추가하여, 에이전트가 단순히 서로를 피하는 것뿐만 아니라 도로의 구조를 따르도록 유도하는 것이 학술적 요구사항에 부합한다. 맵 정보를 통합하지 않을 경우, 모델은 물리적으로 불가능한 경로(예: 화단을 가로지르는 경로)를 예측할 가능성이 높으며, 이는 안전 지표 검증 단계에서 심각한 오류로 이어질 것이다.13

## ---

**3\. 물리적 정합성: 호모그래피 변환의 필수성과 구현 전략**

### **3.1 픽셀 공간의 한계와 미터법 변환의 당위성**

SDD 데이터셋은 드론에서 촬영된 Top-down 뷰를 제공하지만, 원본 데이터는 픽셀 좌표계 $(u, v)$로 제공된다. 제안서에서 언급된 **호모그래피(Homography) 변환**은 선택 사항이 아니라, 물리적으로 타당한 예측을 위한 절대적인 전제 조건이다. 픽셀 공간에서의 거리는 카메라의 렌즈 왜곡과 원근 효과(Perspective Effect)로 인해 실제 물리적 거리와 비선형적인 관계를 갖는다. 예를 들어, 이미지 가장자리의 10픽셀 이동은 이미지 중심부의 10픽셀 이동과 전혀 다른 속도와 가속도를 의미할 수 있다.15

물리학 기반의 운동 방정식(예: 등가속도 운동, 자전거 모델)을 신경망이 학습하기 위해서는 입력 데이터가 유클리드 공간(Metric Space)에 존재해야 한다. 따라서 픽셀 좌표를 미터 단위의 월드 좌표 $(x, y)$로 변환하는 전처리 과정은 필수적이다. 이를 통해 모델은 '픽셀 변화량'이 아닌 실제 '속도($m/s$)'와 '가속도($m/s^2$)'를 학습하게 되며, 이는 후술할 안전 지표 검증의 신뢰성을 담보하는 기초가 된다.

### **3.2 Death Circle 데이터셋의 스케일링 이슈**

연구 자료에 따르면, SDD 데이터셋의 일부 장면, 특히 Death Circle의 경우 픽셀-미터 변환 비율(Scale Factor)이 정확하게 명시되지 않은 경우가 있어 주의가 필요하다. 일반적인 구현에서는 이미지 내의 참조 객체(예: 차량의 길이, 도로 폭, 원형 교차로의 지름)를 통해 스케일 팩터를 역산하거나, Trajectron++와 같은 기존 오픈소스 프로젝트에서 제공하는 사전 계산된 호모그래피 행렬을 활용해야 한다.17

또한, 호모그래피 변환 후에는 에이전트의 속도 분포를 시각화하여 검증하는 절차가 반드시 포함되어야 한다. 예를 들어, 보행자의 평균 속도가 1.2\~1.5m/s 범위 내에 있는지, 차량의 속도가 비상식적으로 높게(예: 100km/h) 계산되지 않는지 확인해야 한다. 만약 변환 행렬이 부정확하다면, 모델은 비현실적인 물리 법칙을 학습하게 되어 일반화 성능이 급격히 저하될 것이다.

## ---

**4\. 안전성 검증 레이어: 단순 정확도를 넘어선 신뢰성 평가**

제안된 프로젝트에서 가장 돋보이는 부분은 **안전 지표(Safety Metrics) 검증**의 도입이다. 이는 2023-2025년 궤적 예측 연구의 핵심 패러다임 전환을 정확히 반영하고 있다. 기존 연구들이 ADE(Average Displacement Error)와 FDE(Final Displacement Error)와 같은 유클리드 거리 오차 최소화에만 집중했다면, 최근 연구들은 "충돌하지 않는 궤적"이 "정확한 평균 궤적"보다 중요함을 강조하고 있다.19

### **4.1 핵심 안전 지표 및 구현 방안**

제안서는 구체적인 안전 지표를 명시하지 않았으나, 본 분석을 통해 다음과 같은 지표들의 구현을 강력히 권장한다.

| 지표 (Metric) | 정의 및 중요성 | 구현 가이드 |
| :---- | :---- | :---- |
| **충돌률 (Collision Rate, CR)** | 테스트 셋에서 예측된 궤적이 타 에이전트와 겹치는 비율. 자율주행의 안전성을 직접적으로 대변함. | 에이전트 유형별로 다른 반경(Radius)을 적용해야 함 (예: 차량 2m, 보행자 0.5m). Shapely 라이브러리를 이용한 폴리곤 교차 검사 권장.21 |
| **충돌 예상 시간 (TTC, Time-to-Collision)** | 현재 속도와 경로를 유지했을 때 충돌까지 남은 시간. | 1.5초 미만의 TTC가 예측될 경우 '위험(Critical)'으로 분류. 단순히 궤적이 겹치는 것을 넘어, 시간적 동기화가 맞는지 확인하는 동적 지표.22 |
| **Joint ADE/FDE (JADE/JFDE)** | 장면(Scene) 전체의 에이전트들이 동시에 올바른 예측을 했는지 평가. | 개별 에이전트의 최적 궤적을 조합하는 것이 아니라, 하나의 샘플링된 시나리오 내에서 모든 에이전트의 궤적 오류를 평균냄. 상호작용의 일관성을 평가하는 최신 지표.20 |

### **4.2 사후 검증을 넘어선 학습 기반 안전 레이어 (Differentiable Safety Layer)**

단순히 예측 후 지표를 계산하는 것을 넘어, \*\*안전 손실 함수(Safety Loss)\*\*를 학습 과정에 통합하는 방안을 고려해야 한다. 예를 들어, 예측된 궤적이 타 에이전트의 미래 위치와 가까워질수록 기하급수적인 페널티를 부여하는 '충돌 방지 손실(Collision Avoidance Loss)'을 추가하거나, 강화학습(RL) 기반의 안전 레이어(Safety Layer)를 통해 물리적으로 불가능하거나 위험한 궤적을 억제하는 방식이 최신 연구 트렌드이다.24 이는 ST-HGNN이 단순히 과거 데이터를 모사하는 것을 넘어, '충돌 회피'라는 사회적 규범을 내재적으로 학습하게 만든다.

## ---

**5\. 구현 계획 보완 및 제안: SOTA 달성을 위한 전략**

### **5.1 비교 모델 (Baselines) 구체화 및 선정 근거**

프로젝트의 성과를 입증하기 위해서는 적절한 비교군 설정이 필수적이다. 단순한 LSTM 모델과의 비교는 현재 시점에서 학술적 가치가 낮다. 다음 모델들을 비교군으로 선정하고 구현할 것을 제안한다.

1. Social-STGCNN 26: 동질적(Homogeneous) GNN의 대표 주자. 이 모델과의 비교를 통해 ST-HGNN의 '이기종성(Heterogeneity)'이 성능 향상에 기여했음을 증명할 수 있다.  
2. Trajectron++ 4: 이기종 CVAE 기반 모델의 표준(Standard). 에이전트의 동역학을 명시적으로 고려하는 모델로, ST-HGNN이 이보다 나은 성능(특히 충돌률 측면에서)을 보인다면 매우 강력한 결과가 된다.  
3. AgentFormer 9: 트랜스포머 기반의 SOTA 모델. GNN 구조가 트랜스포머의 평탄화된 어텐션보다 Death Circle과 같은 밀집 환경에서 어떤 효율성이나 정확도 이점을 가지는지 분석하는 데 중요하다.  
4. VISTA 27: 2025년 기준 최신 SOTA 중 하나로, 목표 지향적(Goal-conditioned) 어텐션을 사용한다. 성능 비교의 상한선(Upper Bound)으로 설정하기 좋다.

### **5.2 시각화 강화 방안: 설명 가능한 AI (XAI)로의 확장**

단순히 예측된 궤적을 지도 위에 그리는 것은 1차원적인 시각화이다. 본 프로젝트의 핵심인 '상호작용'을 입증하기 위해서는 **어텐션 가중치(Attention Weights)의 시각화**가 필수적이다.28

* **동적 상호작용 맵 (Dynamic Interaction Map)**: 특정 시점에 대상 차량(Target Agent)이 어떤 주변 에이전트에게 높은 가중치를 두고 있는지 시각화한다. 예를 들어, Death Circle 진입 시 차량이 진입하려는 자전거에게 높은 어텐션을 주는 것을 히트맵(Heatmap)이나 엣지의 굵기로 표현한다면, 모델이 사회적 상호작용을 학습했음을 정성적으로 증명할 수 있다. 이를 위해 BertViz와 유사한 그래프 어텐션 시각화 도구 또는 DyGETViz와 같은 동적 그래프 시각화 라이브러리의 활용을 추천한다.30  
* **불확실성 시각화**: CVAE나 생성형 모델을 사용할 경우, 예측된 20개의 궤적 샘플을 모두 그려 분포의 분산(Variance)을 보여주거나, 확률적 타원(Probabilistic Ellipse)을 통해 모델의 확신도를 시각화해야 한다.

### **5.3 생성형 모델의 도입 검토**

현재 제안서는 결정론적(Deterministic) 예측인지 확률론적(Stochastic) 예측인지 명시하지 않았다. 그러나 궤적 예측은 본질적으로 불확실성을 내포하므로(Multimodality), 단일 경로 예측은 한계가 있다. ST-HGNN의 디코더 부분에 **CVAE(Conditional Variational Autoencoder)** 또는 최근 급부상하는 \*\*확산 모델(Diffusion Model)\*\*을 결합하여 다중 경로(Multi-modal) 예측을 수행하는 것이 강력히 권장된다.32 이는 특히 Death Circle과 같이 다양한 주행 경로 선택지가 존재하는 환경에서 ADE/FDE 성능을 비약적으로 높일 수 있는 방법이다.

## ---

**6\. 결론**

제안된 'SDD Death Circle 기반 이기종 에이전트 상호작용 예측 프로젝트'는 **ST-HGNN, 호모그래피 변환, 안전 지표 검증**이라는 탄탄한 삼각 구도를 갖추고 있어 연구 방향성이 매우 우수하다. 이는 자율주행 연구 커뮤니티가 지향하는 '안전하고 설명 가능한 사회적 내비게이션'의 목표와 정확히 일치한다.

프로젝트의 성공적인 수행을 위해, 본 보고서는 **(1) 맵 정보(Lane Graph)의 통합을 통한 물리적 제약 강화, (2) 충돌률 및 Joint Metrics와 같은 현대적 안전 지표의 도입, (3) Trajectron++ 및 AgentFormer와 같은 강력한 베이스라인과의 비교 검증, (4) 어텐션 가중치 시각화를 통한 설명력 확보**를 핵심 보완 사항으로 제안한다. 이러한 요소들이 체계적으로 구현된다면, 본 프로젝트는 단순한 모델 구현을 넘어 복잡한 도심 환경에서의 에이전트 상호작용 메커니즘을 규명하는 수준 높은 연구 성과로 이어질 것으로 기대된다.

#### **참고 자료**

1. \[2510.03776\] Trajectory prediction for heterogeneous agents: A performance analysis on small and imbalanced datasets \- arXiv, 12월 14, 2025에 액세스, [https://www.arxiv.org/abs/2510.03776](https://www.arxiv.org/abs/2510.03776)  
2. TraPHic: Trajectory Prediction in Dense and Heterogeneous Traffic Using Weighted Interactions \- CVF Open Access, 12월 14, 2025에 액세스, [https://openaccess.thecvf.com/content\_CVPR\_2019/papers/Chandra\_TraPHic\_Trajectory\_Prediction\_in\_Dense\_and\_Heterogeneous\_Traffic\_Using\_Weighted\_CVPR\_2019\_paper.pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chandra_TraPHic_Trajectory_Prediction_in_Dense_and_Heterogeneous_Traffic_Using_Weighted_CVPR_2019_paper.pdf)  
3. Master's Thesis \- OPUS, 12월 14, 2025에 액세스, [https://opus4.kobv.de/opus4-haw/files/5340/I002211394\_Thesis.pdf](https://opus4.kobv.de/opus4-haw/files/5340/I002211394_Thesis.pdf)  
4. Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data \- European Computer Vision Association, 12월 14, 2025에 액세스, [https://www.ecva.net/papers/eccv\_2020/papers\_ECCV/papers/123630664.pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630664.pdf)  
5. Trajectory prediction for heterogeneous agents: A performance analysis on small and imbalanced datasets \- ResearchGate, 12월 14, 2025에 액세스, [https://www.researchgate.net/publication/396249897\_Trajectory\_prediction\_for\_heterogeneous\_agents\_A\_performance\_analysis\_on\_small\_and\_imbalanced\_datasets](https://www.researchgate.net/publication/396249897_Trajectory_prediction_for_heterogeneous_agents_A_performance_analysis_on_small_and_imbalanced_datasets)  
6. TrajGNAS: Heterogeneous Multiagent Trajectory Prediction Based on a Graph Neural Architecture Search \- CVF Open Access, 12월 14, 2025에 액세스, [https://openaccess.thecvf.com/content/CVPR2025W/WAD/papers/Xu\_TrajGNAS\_Heterogeneous\_Multiagent\_Trajectory\_Prediction\_Based\_on\_a\_Graph\_Neural\_CVPRW\_2025\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025W/WAD/papers/Xu_TrajGNAS_Heterogeneous_Multiagent_Trajectory_Prediction_Based_on_a_Graph_Neural_CVPRW_2025_paper.pdf)  
7. Heterogeneous Graph Neural Network for WiFi RSSI-Based Indoor Floor Classification, 12월 14, 2025에 액세스, [https://www.mdpi.com/2079-9292/14/24/4845](https://www.mdpi.com/2079-9292/14/24/4845)  
8. Trajectory Prediction for Heterogeneous Agents: A Performance Analysis on Small and Imbalanced Datasets | DARKO, 12월 14, 2025에 액세스, [https://darko-project.eu/wp-content/uploads/papers/2024/Trajectory\_Prediction\_for\_Heterogeneous\_Agents\_A\_Performance\_Analysis\_on\_Small\_and\_Imbalanced\_Datasets.pdf](https://darko-project.eu/wp-content/uploads/papers/2024/Trajectory_Prediction_for_Heterogeneous_Agents_A_Performance_Analysis_on_Small_and_Imbalanced_Datasets.pdf)  
9. AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting, 12월 14, 2025에 액세스, [https://www.researchgate.net/publication/358994093\_AgentFormer\_Agent-Aware\_Transformers\_for\_Socio-Temporal\_Multi-Agent\_Forecasting](https://www.researchgate.net/publication/358994093_AgentFormer_Agent-Aware_Transformers_for_Socio-Temporal_Multi-Agent_Forecasting)  
10. Pedestrian Trajectory Prediction Based on Dual Social Graph Attention Network \- MDPI, 12월 14, 2025에 액세스, [https://www.mdpi.com/2076-3417/15/8/4285](https://www.mdpi.com/2076-3417/15/8/4285)  
11. Evolve GAT — A dynamic graph attention model | by Torstein Eliassen | Stanford CS224W, 12월 14, 2025에 액세스, [https://medium.com/stanford-cs224w/evolve-gat-a-dynamic-graph-attention-model-d3a416bb7c33](https://medium.com/stanford-cs224w/evolve-gat-a-dynamic-graph-attention-model-d3a416bb7c33)  
12. \[Literature Review\] GraphAD: Interaction Scene Graph for End-to-end Autonomous Driving, 12월 14, 2025에 액세스, [https://www.themoonlight.io/en/review/graphad-interaction-scene-graph-for-end-to-end-autonomous-driving](https://www.themoonlight.io/en/review/graphad-interaction-scene-graph-for-end-to-end-autonomous-driving)  
13. GraphPilot: Grounded Scene Graph Conditioning for Language-Based Autonomous Driving \- arXiv, 12월 14, 2025에 액세스, [https://arxiv.org/html/2511.11266v1](https://arxiv.org/html/2511.11266v1)  
14. arXiv:2201.07189v1 \[cs.CV\] 18 Jan 2022, 12월 14, 2025에 액세스, [https://arxiv.org/pdf/2201.07189](https://arxiv.org/pdf/2201.07189)  
15. A Review of Homography Estimation: Advances and Challenges \- MDPI, 12월 14, 2025에 액세스, [https://www.mdpi.com/2079-9292/12/24/4977](https://www.mdpi.com/2079-9292/12/24/4977)  
16. Homography Estimation∗, 12월 14, 2025에 액세스, [https://cseweb.ucsd.edu/classes/wi07/cse252a/homography\_estimation/homography\_estimation.pdf](https://cseweb.ucsd.edu/classes/wi07/cse252a/homography_estimation/homography_estimation.pdf)  
17. MapEncoding for ETH set · Issue \#14 · StanfordASL/Trajectron-plus-plus \- GitHub, 12월 14, 2025에 액세스, [https://github.com/StanfordASL/Trajectron-plus-plus/issues/14](https://github.com/StanfordASL/Trajectron-plus-plus/issues/14)  
18. Motion Style Transfer: Modular Low-Rank Adaptation for Deep Motion Forecasting (Supplementary Material), 12월 14, 2025에 액세스, [https://proceedings.mlr.press/v205/kothari23a/kothari23a-supp.pdf](https://proceedings.mlr.press/v205/kothari23a/kothari23a-supp.pdf)  
19. Risk-Aware Vehicle Trajectory Prediction Under Safety-Critical Scenarios \- ResearchGate, 12월 14, 2025에 액세스, [https://www.researchgate.net/publication/388091201\_Risk-Aware\_Vehicle\_Trajectory\_Prediction\_Under\_Safety-Critical\_Scenarios](https://www.researchgate.net/publication/388091201_Risk-Aware_Vehicle_Trajectory_Prediction_Under_Safety-Critical_Scenarios)  
20. Joint Metrics Matter: A Better Standard for Trajectory Forecasting \- CVF Open Access, 12월 14, 2025에 액세스, [https://openaccess.thecvf.com/content/ICCV2023/papers/Weng\_Joint\_Metrics\_Matter\_A\_Better\_Standard\_for\_Trajectory\_Forecasting\_ICCV\_2023\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.pdf)  
21. Misbehavior Detection in Uncertainty Aware Object Level Cooperative Perception for Motion Planning \- ROSA P, 12월 14, 2025에 액세스, [https://rosap.ntl.bts.gov/view/dot/87356/dot\_87356\_DS1.pdf](https://rosap.ntl.bts.gov/view/dot/87356/dot_87356_DS1.pdf)  
22. AI-enabled Interaction-aware Active Safety Analysis with Vehicle Dynamics \- arXiv, 12월 14, 2025에 액세스, [https://arxiv.org/html/2505.00322v1](https://arxiv.org/html/2505.00322v1)  
23. ravesandstorm/Vehicle-TTC-Calculation: Warning model with TTC Estimation using Yolov8, single forward pass \- GitHub, 12월 14, 2025에 액세스, [https://github.com/ravesandstorm/Vehicle-TTC-Calculation](https://github.com/ravesandstorm/Vehicle-TTC-Calculation)  
24. SA-TP 2 ^{2} : A Safety-Aware Trajectory Prediction and Planning Model for Autonomous Driving \- ResearchGate, 12월 14, 2025에 액세스, [https://www.researchgate.net/publication/394781050\_SA-TP2\_A\_Safety-Aware\_Trajectory\_Prediction\_and\_Planning\_Model\_for\_Autonomous\_Driving](https://www.researchgate.net/publication/394781050_SA-TP2_A_Safety-Aware_Trajectory_Prediction_and_Planning_Model_for_Autonomous_Driving)  
25. Enforcing Hard Constraints with Soft Barriers: Safe Reinforcement Learning in Unknown Stochastic Environments, 12월 14, 2025에 액세스, [https://proceedings.mlr.press/v202/wang23as/wang23as.pdf](https://proceedings.mlr.press/v202/wang23as/wang23as.pdf)  
26. Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction \- ResearchGate, 12월 14, 2025에 액세스, [https://www.researchgate.net/publication/339551042\_Social-STGCNN\_A\_Social\_Spatio-Temporal\_Graph\_Convolutional\_Neural\_Network\_for\_Human\_Trajectory\_Prediction](https://www.researchgate.net/publication/339551042_Social-STGCNN_A_Social_Spatio-Temporal_Graph_Convolutional_Neural_Network_for_Human_Trajectory_Prediction)  
27. VISTA: A Vision and Intent-Aware Social Attention Framework for Multi-Agent Trajectory Prediction \- arXiv, 12월 14, 2025에 액세스, [https://arxiv.org/html/2511.10203v1](https://arxiv.org/html/2511.10203v1)  
28. Pedestrian trajectory prediction method based on graph neural networks. \- GitHub, 12월 14, 2025에 액세스, [https://github.com/Chenwangxing/Review-of-PTP-Based-on-GNNs](https://github.com/Chenwangxing/Review-of-PTP-Based-on-GNNs)  
29. What are some common techniques for visualizing attention weights? \- Infermatic.ai, 12월 14, 2025에 액세스, [https://infermatic.ai/ask/?question=What%20are%20some%20common%20techniques%20for%20visualizing%20attention%20weights?](https://infermatic.ai/ask/?question=What+are+some+common+techniques+for+visualizing+attention+weights?)  
30. Ahren09/dygetviz: Dynamic Graph Embedding Trajectory Visualization \- Interactive python tool for exploring temporal graph evolution \- GitHub, 12월 14, 2025에 액세스, [https://github.com/Ahren09/dygetviz](https://github.com/Ahren09/dygetviz)  
31. BertViz: Visualize Attention in NLP Models (BERT, GPT2, BART, etc.) \- GitHub, 12월 14, 2025에 액세스, [https://github.com/jessevig/bertviz](https://github.com/jessevig/bertviz)  
32. \[Quick Review\] Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion \- Liner, 12월 14, 2025에 액세스, [https://liner.com/review/stochastic-trajectory-prediction-via-motion-indeterminacy-diffusion](https://liner.com/review/stochastic-trajectory-prediction-via-motion-indeterminacy-diffusion)  
33. MAD-Traj: Multi-modal Attention-based Diffu- sion Model for Pedestrian Trajectory Prediction, 12월 14, 2025에 액세스, [https://vbn.aau.dk/ws/files/538310014/MAD\_Traj.pdf](https://vbn.aau.dk/ws/files/538310014/MAD_Traj.pdf)