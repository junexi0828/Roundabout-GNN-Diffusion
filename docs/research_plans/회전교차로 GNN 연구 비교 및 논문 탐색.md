# **회전교차로(Roundabout) 내 자율주행 의사결정을 위한 씬 그래프 및 GNN 기반 상호작용 예측 연구: 악천후 강건성 연구와의 경쟁력 비교 및 실행 전략**

## **1\. 서론: 자율주행 연구의 패러다임 전환과 소논문 주제 선정의 중요성**

현대 자율주행(Autonomous Driving, AD) 기술은 단순한 인지(Perception) 단계를 넘어 인지된 객체들의 행동을 예측하고(Prediction), 복잡한 사회적 상호작용을 이해하여 안전한 주행 경로를 생성(Planning)하는 '인지적 추론(Cognitive Reasoning)' 단계로 진화하고 있습니다. 초기 자율주행 연구가 카메라, 라이다(LiDAR), 레이더(Radar) 센서를 활용하여 악천후나 조도 변화와 같은 환경적 제약 속에서도 객체를 정확히 검출하는 '강건성(Robustness)' 확보에 주력했다면, 최근의 학술적 흐름은 검출된 객체들 간의 관계를 모델링하고, 그들의 잠재적 의도를 파악하여 사고를 미연에 방지하는 '상호작용 예측(Interaction Prediction)'으로 그 무게중심이 이동하고 있습니다.

특히 회전교차로(Roundabout)는 자율주행 차량에 있어 가장 난이도가 높은 주행 시나리오 중 하나로 꼽힙니다. 신호등에 의해 통제되는 일반 교차로와 달리, 회전교차로는 진입 차량과 회전 차량 간의 양보(Yield)와 진입(Merge)이라는 고도의 눈치 싸움, 즉 암묵적인 상호작용 규칙이 지배하는 공간이기 때문입니다.1 이러한 환경에서는 단순히 물리적인 거리나 속도만으로 주변 차량의 행동을 예측하는 데 한계가 있으며, "저 차량이 나를 보고 속도를 줄이는가?" 혹은 "저 차량은 내가 진입할 공간을 내어줄 의사가 있는가?"와 같은 관계적 추론이 필수적입니다.

본 보고서는 학부생 또는 석사 과정 연구생이 작성할 소논문(Minor Thesis)의 주제로서, 기존의 전통적인 주제인 \*\*'악천후 강건성(Robustness in Adverse Weather)'\*\*과 비교하여, **'씬 그래프(Scene Graph)와 그래프 신경망(GNN)을 활용한 회전교차로 내 상호작용 및 주행 판단'** 연구가 가지는 학술적 가치와 경쟁력을 심층적으로 분석합니다. 나아가, 후자의 주제를 채택했을 때 성공적인 연구 수행을 위해 필요한 핵심 문헌(INTERACTION Dataset, Social-STGCNN 등)을 고찰하고, 데이터셋 구축부터 모델링, 가상 결과 도출에 이르는 구체적이고 실행 가능한 연구 계획을 제안합니다.

이 보고서의 핵심 주장은 **'씬 그래프와 GNN을 활용한 상호작용 예측'** 주제가 '악천후 강건성' 주제보다 학생 연구자에게 있어 **데이터 접근성, 연산 자원의 효율성, 그리고 학술적 참신성(Novelty)** 측면에서 월등히 높은 경쟁력을 가진다는 것입니다. 악천후 연구가 거대 자본과 방대한 센서 데이터를 요구하는 '데이터 엔지니어링' 성격이 강해진 반면, 상호작용 예측 연구는 그래프 이론과 딥러닝 아키텍처 설계를 통한 '논리적 모델링' 능력을 보여주기에 최적화된 분야이기 때문입니다.

## ---

**2\. 연구 주제의 경쟁력 비교 분석: 악천후 강건성 vs. 씬 그래프 기반 상호작용**

학술 연구, 특히 학위 논문이나 소논문의 주제를 선정할 때는 단순히 기술적 트렌드만을 좇는 것이 아니라, 연구의 **실현 가능성(Feasibility)**, **차별성(Novelty)**, 그리고 \*\*학술적 기여도(Contribution)\*\*를 종합적으로 고려해야 합니다.

### **2.1 기존 주제 분석: 악천후 강건성 (Robustness in Adverse Weather)**

악천후 강건성 연구는 비, 눈, 안개, 야간 등 시각적 악조건 속에서도 객체 검출(Object Detection)이나 의미론적 분할(Semantic Segmentation)의 성능을 유지하는 것을 목표로 합니다.

#### **2.1.1 연구의 현황과 한계 (The Perception Plateau)**

이 분야는 자율주행의 태동기와 함께 시작되어 현재 상당히 성숙한 단계(Maturation Phase)에 진입했습니다. Waymo, Tesla와 같은 선도 기업들은 수백만 마일의 실제 주행 데이터를 통해 악천후 데이터를 축적하고 있으며, 학계에서는 다음과 같은 기술들이 주류를 이룹니다.

* **센서 퓨전(Sensor Fusion):** 카메라의 시각 정보를 라이다의 깊이 정보나 레이더의 속도 정보와 결합하여 악천후로 인한 노이즈를 상쇄하는 방식입니다. 이는 고가의 센서 장비와 이를 처리하기 위한 막대한 컴퓨팅 파워를 요구합니다.  
* **도메인 적응(Domain Adaptation):** 맑은 날씨의 데이터를 비 오는 날씨 스타일로 변환(Style Transfer)하여 학습 데이터를 증강하는 방식입니다. CycleGAN 등의 생성 모델이 주로 사용됩니다.

#### **2.1.2 소논문으로서의 경쟁력 평가**

학생 연구자 입장에서 이 주제는 다음과 같은 치명적인 약점을 가집니다.

1. **데이터 희소성:** nuScenes 2이나 Waymo Open Dataset 4에 'Rain' 태그가 포함된 데이터가 존재하지만, 폭우나 폭설과 같은 극한 상황(Edge Case)에 대한 고품질 데이터는 여전히 부족합니다. 연구자가 직접 데이터를 수집하기에는 위험 부담과 비용이 너무 큽니다.  
2. **성능 향상의 포화:** 이미 SOTA(State-of-the-Art) 모델들의 성능이 매우 높아, 학부 수준의 연구에서 획기적인 성능 향상(예: mAP 5% 이상 증가)을 보여주기가 어렵습니다. 0.5\~1%의 미미한 성능 향상은 논문의 설득력을 떨어뜨립니다.  
3. **장비 의존성:** 3D Point Cloud와 고해상도 이미지를 동시에 처리하는 멀티 모달 퓨전 모델은 고사양 GPU(A100, V100 등) 클러스터를 필요로 하는 경우가 많아, 개인 연구 환경에서는 실험 자체가 불가능할 수 있습니다.

### **2.2 제안 주제 분석: 씬 그래프와 GNN을 활용한 회전교차로 상호작용 (The Cognitive Frontier)**

이 주제는 센서로부터 얻은 객체 정보를 **그래프(Graph)** 형태로 추상화하고, \*\*그래프 신경망(GNN)\*\*을 통해 객체 간의 관계를 학습하여 미래 경로를 예측하거나 주행 의사결정(진입/양보)을 내리는 연구입니다.

#### **2.2.1 씬 그래프(Scene Graph)와 GNN의 학술적 가치**

* **구조적 추론(Structured Reasoning):** 도로는 격자(Grid)가 아닌 그래프 구조에 가깝습니다. 차량(Node)들은 도로망(Edge)을 따라 이동하며 서로 영향을 주고받습니다. 씬 그래프는 이러한 위상학적(Topological) 구조를 명시적으로 모델링할 수 있는 최적의 도구입니다.5  
* **설명 가능성(Explainability):** 딥러닝의 고질적인 문제인 '블랙박스' 특성을 완화할 수 있습니다. GNN의 어텐션(Attention) 가중치를 시각화하면, 자율주행차가 회전교차로에 진입할 때 "어떤 차량을 주시하고 있는지"를 시각적으로 보여줄 수 있습니다.7 이는 논문의 질적 평가(Qualitative Result)에서 매우 강력한 무기가 됩니다.

#### **2.2.2 소논문으로서의 경쟁력 평가**

1. **높은 참신성:** 자율주행 분야에서 동적 씬 그래프(Dynamic Scene Graph)를 활용한 연구는 2020년 이후 본격화된 최신 트렌드입니다.8 특히 회전교차로의 '딜레마 존(Dilemma Zone)' 해결에 GNN을 적용하는 것은 여전히 활발한 연구가 필요한 '블루오션'입니다.1  
2. **데이터 접근성 및 효율성:** **INTERACTION Dataset** 10과 같이 드론으로 촬영되어 사각지대가 없고 상호작용이 풍부한 데이터셋이 공개되어 있습니다. 또한, 이미지 픽셀 전체를 처리하는 CNN과 달리, 객체의 좌표(Coordinate)만을 입력으로 받는 GNN은 연산량이 현저히 적어 일반적인 연구실 워크스테이션이나 고사양 노트북으로도 충분히 학습과 실험이 가능합니다.  
3. **명확한 기여점:** 단순히 "정확도를 높였다"는 결과뿐만 아니라, "상호작용의 원리를 규명했다"는 논리적 기여가 가능합니다. 예를 들어, "GNN 모델이 회전교차로 내 차량 간의 거리가 가까워질수록 엣지(Edge)의 가중치를 높게 학습함을 확인했다"는 식의 분석은 소논문의 가치를 크게 높여줍니다.

### **2.3 종합 비교표**

| 비교 항목 | 주제 A: 악천후 강건성 (Adverse Weather) | 주제 B: 씬 그래프 & GNN 상호작용 (Recommended) |
| :---- | :---- | :---- |
| **연구의 초점** | 감각/인지 (Perception) | 인지/추론 (Cognition/Reasoning) |
| **핵심 기술** | CNN, Sensor Fusion, GAN, Denoising | GNN, GAT, LSTM, Scene Graph Generation |
| **데이터 요구사항** | 대용량 이미지/라이다, 다양한 날씨 데이터 (수집 어려움) | 객체 궤적(Trajectory) 데이터, HD Map (공개 데이터 충분) |
| **연산 자원** | 매우 높음 (Multi-GPU 권장) | 낮음\~중간 (Single GPU 가능) |
| **학술적 참신성** | 낮음 (성숙기, 미세 튜닝 위주) | 높음 (성장기, 구조적 모델링 제안 용이) |
| **결과물의 형태** | mAP, IoU 등 수치적 성능 향상 | 경로 예측 오차 감소 \+ **상호작용 시각화(Attention Map)** |
| **추천 대상** | 하드웨어/신호처리 중심 연구자 | **AI 모델링/알고리즘/로보틱스 중심 연구자** |

**결론적으로, 소논문으로서의 경쟁력은 주제 B(씬 그래프 & GNN)가 압도적으로 우세합니다.** 이는 학생 연구자가 '데이터의 양'이 아닌 '아이디어의 질'로 승부할 수 있는 영역이며, 결과물의 시각적 설득력 또한 뛰어나기 때문입니다.

## ---

**3\. 이론적 배경 및 핵심 기술 요소**

본 연구를 수행하기 위해서는 씬 그래프와 GNN, 그리고 이를 자율주행에 적용하는 메커니즘에 대한 깊은 이해가 필요합니다.

### **3.1 자율주행에서의 씬 그래프 (Scene Graph)**

일반적으로 지식 그래프(Knowledge Graph)가 "정지 표지판은 멈춰야 한다"와 같은 정적인 규칙과 사실을 저장하는 온톨로지(Ontology)라면 11, \*\*씬 그래프(Scene Graph)\*\*는 매 순간 변화하는 도로의 상황을 그래프로 표현한 인스턴스입니다.6

* **노드(Nodes, $V$):** 도로 위의 객체들을 나타냅니다.  
  * *동적 노드:* 자차(Ego-vehicle), 주변 차량(Target Vehicles), 보행자.  
  * *정적 노드:* 차선(Lane Centerline), 정지선, 횡단보도.  
* **엣지(Edges, $E$):** 객체 간의 관계를 나타냅니다.  
  * *공간적 엣지:* "차량 A는 차량 B의 **앞에** 있다(Front-of)", "차량 A는 차선 L **위에** 있다(On)".  
  * *의미론적 엣지:* "차량 A는 차량 B에게 **양보해야 한다**(Yield-to)", "차량 A는 보행자 P를 **주시하고 있다**(Attend-to)".

회전교차로 연구에서는 이 씬 그래프를 **상호작용 씬 그래프(Interaction Scene Graph, ISG)** 8로 확장하여 정의합니다. ISG는 동적 객체 간의 관계뿐만 아니라, 객체와 지도(Map) 요소 간의 관계를 통합적으로 표현하여, 차량이 도로의 기하학적 구조(곡률)에 맞춰 주행하면서 동시에 주변 차량과 충돌을 회피하는 복합적인 추론을 가능케 합니다.

### **3.2 그래프 신경망 (Graph Neural Networks, GNN)**

GNN은 그래프 구조 데이터에서 노드와 엣지의 정보를 학습하는 딥러닝 모델입니다. 자율주행 경로 예측에서 GNN은 주로 **메시지 전달(Message Passing)** 방식을 통해 주변 차량의 정보를 자차의 특징 벡터(Feature Vector)에 통합합니다.

#### **3.2.1 GNN의 작동 원리 (Trajectory Prediction 관점)**

1. **임베딩(Embedding):** 각 차량 $i$의 과거 궤적 $(x\_t, y\_t, v\_t)$을 LSTM이나 1D-CNN을 통과시켜 고정된 길이의 특징 벡터 $h\_i$로 변환합니다.  
2. 메시지 생성 및 집계(Aggregation): 차량 $i$의 이웃 차량들($j \\in \\mathcal{N}(i)$)로부터 메시지를 받습니다.

   $$m\_{ij} \= \\phi(h\_i, h\_j, e\_{ij})$$

   여기서 $e\_{ij}$는 두 차량 간의 상대적 거리나 관계 정보를 담은 엣지 속성입니다.  
3. 상태 업데이트: 집계된 메시지를 이용하여 차량 $i$의 상태를 업데이트합니다. 이 업데이트된 상태 벡터 $h\_i'$는 "주변 차량의 상황을 인지한 후의 자차 상태"를 의미합니다.

   $$h\_i' \= \\gamma(h\_i, \\sum\_{j} m\_{ij})$$

#### **3.2.2 주요 GNN 변형 모델**

* **GCN (Graph Convolutional Network):** 고정된 가중치로 이웃 노드의 정보를 합칩니다. 하지만 모든 이웃 차량을 동일한 중요도로 처리한다는 단점이 있습니다.15  
* **GAT (Graph Attention Network):** \*\*어텐션 메커니즘(Attention Mechanism)\*\*을 도입하여, 특정 이웃(예: 충돌 위험이 높은 차량)에게 더 높은 가중치를 부여합니다.7 회전교차로에서는 진입하려는 차량 입장에서 '이미 회전 중인 차량'이 '내 뒤따라오는 차량'보다 훨씬 중요하므로, GAT가 필수적입니다.

### **3.3 회전교차로의 특수성과 딜레마 존 (Dilemma Zone)**

회전교차로는 **딜레마 존(Dilemma Zone, DZ)** 현상이 빈번하게 발생하는 구역입니다.1 DZ는 운전자가 교차로에 진입할지 멈출지 결정하기 어려운 시공간적 영역을 의미합니다.

* GNN 기반 모델은 이 DZ 상황에서 진입 차량과 회전 차량 사이의 **잠재적 충돌 확률**을 그래프 엣지의 속성으로 모델링함으로써, 단순한 물리적 거리 기반 판단보다 훨씬 정교한 예측을 수행할 수 있습니다.

## ---

**4\. 핵심 문헌 및 데이터셋 심층 분석**

성공적인 연구를 위해서는 적절한 선행 연구 분석과 데이터셋 선정이 필수적입니다.

### **4.1 데이터셋 선정: INTERACTION Dataset**

많은 학생들이 자율주행 연구에 **nuScenes** 2를 사용하려 하지만, 회전교차로 상호작용 연구에는 적합하지 않을 수 있습니다. nuScenes는 주로 직진 위주의 도심 주행 데이터이며, 차량 간의 복잡한 끼어들기나 양보 상황이 상대적으로 적습니다. 또한 센서 데이터(LiDAR, Camera) 위주라 궤적 추출 전처리가 번거롭습니다.

본 연구에는 **INTERACTION Dataset** 10이 가장 적합합니다.

* **특화된 시나리오:** 이 데이터셋은 이름 그대로 '상호작용(Interaction)'에 초점을 맞추었으며, 미국, 중국, 독일 등 다양한 국가의 **회전교차로(Roundabout)**, 비신호 교차로, 합류 구간(Merging) 데이터를 전문적으로 제공합니다.  
* **드론 데이터:** 드론을 통해 상공에서 촬영하여 사각지대가(occlusion) 전혀 없는 완벽한 궤적 데이터를 제공합니다. 이는 '인지' 오차를 배제하고 순수하게 '예측' 알고리즘의 성능을 검증하기에 최적입니다.  
* **HD Map 제공:** Lanelet2 포맷의 고정밀 지도를 제공하여, 차량과 차선 간의 관계(GNN의 정적 노드)를 구성하기 용이합니다.

### **4.2 핵심 관련 논문 (Key Papers)**

연구의 논리를 전개하기 위해 반드시 인용하고 분석해야 할 논문들입니다.

1. Social-LSTM (Alahi et al., CVPR 2016\) 19:  
   * *내용:* 보행자 궤적 예측에 LSTM을 사용하고, 주변 이웃의 정보를 'Social Pooling'이라는 격자 기반 방식으로 통합했습니다.  
   * *한계:* 격자 방식은 회전교차로처럼 곡선이 많은 환경에서 공간적 관계를 제대로 표현하지 못합니다. 본 연구의 비교 대상(Baseline)으로 적합합니다.  
2. Social-GAN (Gupta et al., CVPR 2018\) 21:  
   * *내용:* GAN을 도입하여 하나의 과거 궤적에 대해 여러 개의 가능한 미래 궤적(Multimodal Prediction)을 생성했습니다.  
   * *의의:* 회전교차로 진입 차량은 '진입'하거나 '양보'하는 두 가지 모드를 가지므로, 이러한 다중 모드 예측이 필수적임을 논증하는 데 사용됩니다.  
3. Social-STGCNN (Mohamed et al., CVPR 2020\) 16:  
   * *내용:* 상호작용을 \*\*시공간 그래프(Spatio-Temporal Graph)\*\*로 모델링했습니다. 기존의 LSTM 대신 GCN과 TCN(Temporal CNN)을 사용하여 연산 속도를 획기적으로 높였습니다.  
   * *활용:* 본 연구의 \*\*기본 모델(Base Model)\*\*로 삼기에 가장 적합합니다. 이 구조를 차량 데이터에 맞게 수정하고, 지도(Map) 정보를 추가하는 방식으로 연구를 확장할 수 있습니다.  
4. VectorNet (Gao et al., CVPR 2020\) 24:  
   * *내용:* HD Map의 차선과 차량의 궤적을 모두 \*\*벡터(Vector)\*\*로 표현하고, 이를 GNN으로 처리했습니다.  
   * *의의:* 래스터 이미지(Raster Image) 대신 벡터 그래프를 사용하는 것이 회전교차로의 복잡한 위상을 표현하는 데 얼마나 효율적인지 설명하는 근거가 됩니다.  
5. GraphAD (Zhang et al., arXiv 2024/IJCAI 2025\) 7:  
   * *내용:* 인지부터 계획까지의 전 과정을 \*\*상호작용 씬 그래프(Interaction Scene Graph)\*\*로 통합했습니다.  
   * *의의:* 본 연구가 지향해야 할 최신 SOTA 방향성을 제시합니다. 동적 씬 그래프(DSG)와 정적 씬 그래프(SSG)의 결합 개념을 차용할 수 있습니다.

## ---

**5\. 구체적 연구 실행 계획 (Research Execution Plan)**

다음은 실제 소논문 작성을 위한 단계별 실행 로드맵입니다.

### **5.1 1단계: 데이터셋 구축 및 전처리 (Data Preprocessing)**

1. **데이터 확보:** INTERACTION Dataset 공식 웹사이트 10에서 Roundabout 카테고리(예: DR\_USA\_Roundabout\_FT)의 데이터를 다운로드합니다. train/val 분할은 제공된 툴킷을 사용합니다.  
2. **그래프 데이터 변환:** Raw CSV 파일(Time, ID, x, y, vx, vy)을 GNN 학습을 위한 그래프 객체로 변환해야 합니다. Python의 NetworkX 25 또는 PyTorch Geometric 라이브러리를 사용합니다.  
   * **노드 특징(Feature) 정의:** 각 시점 $t$의 차량 $i$에 대해 $v\_i^t \= \[x, y, v\_x, v\_y, \\text{heading}, \\text{width}, \\text{length}\]$. 좌표는 절대 좌표가 아닌, 타겟 차량 중심의 상대 좌표로 정규화(Normalization)하는 것이 성능에 유리합니다.26  
   * **인접 행렬(Adjacency Matrix) $A$ 구축:**  
     * *거리 기반 연결:* 특정 반경(예: 30m) 내에 있는 차량끼리 엣지를 연결합니다.  
     * *위상 기반 연결(심화):* Lanelet2 지도를 파싱하여, 동일한 차선이나 진로가 겹치는 차선에 있는 차량끼리 연결합니다. 이는 모델이 물리적으로 멀리 있어도 상호작용이 필요한 차량(예: 고속으로 다가오는 회전 차량)을 인지하게 돕습니다.

### **5.2 2단계: GNN 기반 모델링 (Modeling Architecture)**

**Encoder-Decoder 구조**를 채택합니다.

1. **Temporal Encoder (시계열 인코더):**  
   * 각 차량의 과거 3초(30프레임, 10Hz 기준) 궤적을 입력받아 LSTM 또는 GRU를 통과시켜 **모션 임베딩(Motion Embedding)** 벡터를 생성합니다. 이는 각 차량의 개인적인 운동 특성을 압축합니다.19  
2. **Interaction Module (상호작용 모듈 \- 핵심):**  
   * 생성된 모션 임베딩들을 그래프의 노드로 배치합니다.  
   * **GAT (Graph Attention Network) 레이어**를 2\~3층 쌓습니다.  
   * $$\\alpha\_{ij} \= \\text{softmax}(\\text{LeakyReLU}(a^T))$$  
   * 이 수식은 차량 $i$가 차량 $j$에게 얼마나 집중(Attention)해야 하는지를 나타내는 가중치 $\\alpha\_{ij}$를 계산합니다. 회전교차로 진입 시 충돌 위험이 높은 차량일수록 높은 $\\alpha$ 값을 가지도록 학습됩니다.  
3. **Decoder (경로 디코더):**  
   * 상호작용 정보가 반영된 최종 특징 벡터를 입력받아 미래 5초(50프레임)의 궤적을 예측합니다.  
   * 단일 경로가 아닌, **이변량 가우스 분포(Bivariate Gaussian Distribution)** 27를 출력하여 불확실성을 표현합니다. 즉, 모델은 $(x, y)$ 좌표뿐만 아니라 분산 $(\\sigma\_x, \\sigma\_y)$과 상관계수 $(\\rho)$를 함께 예측하여, 예측의 신뢰도를 제공합니다.

### **5.3 3단계: 가상 결과 도출 및 시각화 (Visualization & Evaluation)**

사용자가 우려한 "결과 조작(faking)" 29이 아닌, 학술적으로 타당한 **"정성적 분석(Qualitative Analysis)"** 전략입니다. 수치적 성능(ADE/FDE)이 SOTA보다 약간 낮더라도, 다음과 같은 분석을 통해 논문의 가치를 입증할 수 있습니다.

1. **어텐션 맵(Attention Map) 시각화:**  
   * Matplotlib과 NetworkX를 사용하여, 회전교차로의 BEV(Bird's Eye View) 이미지 위에 차량(노드)과 상호작용(엣지)을 그립니다.31  
   * **핵심 전략:** GAT 레이어에서 추출한 어텐션 가중치 $\\alpha\_{ij}$에 비례하여 **엣지의 두께나 색상 투명도**를 조절합니다.  
   * *시나리오:* 자차(진입 차량)가 회전 중인 차량을 기다리며 정지하는 순간, 두 차량을 연결하는 엣지가 굵고 진하게 표시된다면, 이는 모델이 "양보해야 할 대상"을 정확히 인지했다는 강력한 증거가 됩니다.  
2. **가상 시나리오 검증 (Synthetic Validation):**  
   * INTERACTION 데이터 외에도, **CARLA 시뮬레이터** 3를 사용하여 통제된 실험을 수행합니다. 예를 들어, 무조건 양보해야 하는 극단적인 상황을 연출하고 모델이 이를 예측하는지 테스트합니다. 이는 실제 데이터의 노이즈를 제거하고 모델의 논리적 결함을 찾는 데 유용한 학술적 방법론입니다(Simulation-to-Real).  
3. **비교 평가 (Quantitative Metrics):**  
   * **ADE (Average Displacement Error):** 전체 예측 시간 동안의 평균 거리 오차.  
   * **FDE (Final Displacement Error):** 마지막 시점(예: 5초 후)의 위치 오차.  
   * **Miss Rate:** 예측 궤적이 실제 궤적과 2m 이상 벗어난 비율.  
   * 단순한 물리 모델(Constant Velocity Model, CVM)과 비교하여 GNN 모델이 곡선 구간에서 얼마나 압도적인 성능을 보이는지 표로 제시합니다.

## ---

**6\. 결론**

본 보고서의 분석 결과, \*\*'씬 그래프와 GNN을 활용한 회전교차로 상호작용 연구'\*\*는 학부생 및 석사 수준의 소논문으로서 최적의 선택지임이 확인되었습니다.

1. **경쟁력:** 거대 자본이 필요한 '악천후 강건성' 연구와 달리, 논리적 모델링과 알고리즘 설계 능력만으로도 충분히 경쟁력 있는 결과물을 낼 수 있습니다.  
2. **학술적 트렌드:** 자율주행 연구의 최전선인 '인지적 추론'과 '설명 가능한 AI(XAI)'의 흐름에 정확히 부합합니다.  
3. **실행 용이성:** INTERACTION Dataset이라는 고품질 공개 데이터와 PyTorch Geometric 등 강력한 오픈소스 라이브러리를 통해 즉각적인 연구 착수가 가능합니다.

연구자는 본 보고서에서 제시한 로드맵에 따라 **INTERACTION 데이터셋 전처리 \-\> GAT 기반 인코더 설계 \-\> 어텐션 맵 시각화**의 순서로 연구를 진행한다면, 기술적 깊이와 시각적 설득력을 모두 갖춘 우수한 논문을 완성할 수 있을 것입니다. 단순한 성능 수치 경쟁을 넘어, "자율주행차가 왜 멈추었는가?"에 대한 답을 그래프로 보여주는 것이 이 연구의 핵심 성공 요인입니다.

#### **참고 자료**

1. Roundabout Dilemma Zone Data Mining and Forecasting with Trajectory Prediction and Graph Neural Networks \- arXiv, 12월 14, 2025에 액세스, [https://arxiv.org/html/2409.00622v1](https://arxiv.org/html/2409.00622v1)  
2. nuScenes devkit tutorial, 12월 14, 2025에 액세스, [https://www.nuscenes.org/tutorials/nuscenes\_tutorial.html](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html)  
3. nuCarla: A nuScenes-Style Bird's-Eye View Perception Dataset for CARLA Simulation, 12월 14, 2025에 액세스, [https://arxiv.org/html/2511.13744v1](https://arxiv.org/html/2511.13744v1)  
4. Download – Waymo Open Dataset, 12월 14, 2025에 액세스, [https://waymo.com/intl/jp/open/download](https://waymo.com/intl/jp/open/download)  
5. T2SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving \- CVF Open Access, 12월 14, 2025에 액세스, [https://openaccess.thecvf.com/content/CVPR2025/papers/Lv\_T2SG\_Traffic\_Topology\_Scene\_Graph\_for\_Topology\_Reasoning\_in\_Autonomous\_CVPR\_2025\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Lv_T2SG_Traffic_Topology_Scene_Graph_for_Topology_Reasoning_in_Autonomous_CVPR_2025_paper.pdf)  
6. Query by Example: Semantic Traffic Scene Retrieval Using LLM-Based Scene Graph Representation \- PubMed Central, 12월 14, 2025에 액세스, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12031543/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12031543/)  
7. GraphAD: Interaction Scene Graph for End-to-end Autonomous Driving \- IJCAI, 12월 14, 2025에 액세스, [https://www.ijcai.org/proceedings/2025/0270.pdf](https://www.ijcai.org/proceedings/2025/0270.pdf)  
8. \[Literature Review\] GraphAD: Interaction Scene Graph for End-to-end Autonomous Driving, 12월 14, 2025에 액세스, [https://www.themoonlight.io/en/review/graphad-interaction-scene-graph-for-end-to-end-autonomous-driving](https://www.themoonlight.io/en/review/graphad-interaction-scene-graph-for-end-to-end-autonomous-driving)  
9. GraphAD: Interaction Scene Graph for End-to-end Autonomous Driving \- arXiv, 12월 14, 2025에 액세스, [https://arxiv.org/html/2403.19098v1](https://arxiv.org/html/2403.19098v1)  
10. Interaction Dataset, 12월 14, 2025에 액세스, [http://interaction-dataset.com/](http://interaction-dataset.com/)  
11. Predicting the Road Ahead: A Knowledge Graph based Foundation Model for Scene Understanding in Autonomous Driving \- ResearchGate, 12월 14, 2025에 액세스, [https://www.researchgate.net/publication/390143346\_Predicting\_the\_Road\_Ahead\_A\_Knowledge\_Graph\_based\_Foundation\_Model\_for\_Scene\_Understanding\_in\_Autonomous\_Driving](https://www.researchgate.net/publication/390143346_Predicting_the_Road_Ahead_A_Knowledge_Graph_based_Foundation_Model_for_Scene_Understanding_in_Autonomous_Driving)  
12. Research on Driving Scenario Knowledge Graphs \- MDPI, 12월 14, 2025에 액세스, [https://www.mdpi.com/2076-3417/14/9/3804](https://www.mdpi.com/2076-3417/14/9/3804)  
13. Relation-based Motion Prediction using Traffic Scene Graphs, 12월 14, 2025에 액세스, [https://www.uni-trier.de/fileadmin/fb2/LDV/Testmaterial/Relation-based\_Motion\_Prediction\_using\_Traffic\_Scene\_Graphs.pdf](https://www.uni-trier.de/fileadmin/fb2/LDV/Testmaterial/Relation-based_Motion_Prediction_using_Traffic_Scene_Graphs.pdf)  
14. GraphAD: Interaction Scene Graph for End-to-end Autonomous Driving | IJCAI, 12월 14, 2025에 액세스, [https://www.ijcai.org/proceedings/2025/270](https://www.ijcai.org/proceedings/2025/270)  
15. Pedestrian trajectory prediction method based on graph neural networks. \- GitHub, 12월 14, 2025에 액세스, [https://github.com/Chenwangxing/Review-of-PTP-Based-on-GNNs](https://github.com/Chenwangxing/Review-of-PTP-Based-on-GNNs)  
16. Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction \- ResearchGate, 12월 14, 2025에 액세스, [https://www.researchgate.net/publication/339551042\_Social-STGCNN\_A\_Social\_Spatio-Temporal\_Graph\_Convolutional\_Neural\_Network\_for\_Human\_Trajectory\_Prediction](https://www.researchgate.net/publication/339551042_Social-STGCNN_A_Social_Spatio-Temporal_Graph_Convolutional_Neural_Network_for_Human_Trajectory_Prediction)  
17. Pedestrian Trajectory Prediction in Crowded Environments Using Social Attention Graph Neural Networks \- MDPI, 12월 14, 2025에 액세스, [https://www.mdpi.com/2076-3417/14/20/9349](https://www.mdpi.com/2076-3417/14/20/9349)  
18. jiachenli94/Awesome-Interaction-Aware-Trajectory-Prediction \- GitHub, 12월 14, 2025에 액세스, [https://github.com/jiachenli94/Awesome-Interaction-Aware-Trajectory-Prediction](https://github.com/jiachenli94/Awesome-Interaction-Aware-Trajectory-Prediction)  
19. Social LSTM: Human Trajectory Prediction in Crowded Spaces \- ResearchGate, 12월 14, 2025에 액세스, [https://www.researchgate.net/publication/311611429\_Social\_LSTM\_Human\_Trajectory\_Prediction\_in\_Crowded\_Spaces](https://www.researchgate.net/publication/311611429_Social_LSTM_Human_Trajectory_Prediction_in_Crowded_Spaces)  
20. Social LSTM: Human Trajectory Prediction in Crowded Spaces \- Stanford Computational Vision and Geometry Lab, 12월 14, 2025에 액세스, [https://cvgl.stanford.edu/papers/CVPR16\_Social\_LSTM.pdf](https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf)  
21. Vehicle trajectory prediction and generation using LSTM models and GANs \- PMC \- NIH, 12월 14, 2025에 액세스, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8248611/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8248611/)  
22. A Survey of Autonomous Driving Trajectory Prediction: Methodologies, Challenges, and Future Prospects \- MDPI, 12월 14, 2025에 액세스, [https://www.mdpi.com/2075-1702/13/9/818](https://www.mdpi.com/2075-1702/13/9/818)  
23. Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction \- CVF Open Access, 12월 14, 2025에 액세스, [https://openaccess.thecvf.com/content\_CVPR\_2020/papers/Mohamed\_Social-STGCNN\_A\_Social\_Spatio-Temporal\_Graph\_Convolutional\_Neural\_Network\_for\_Human\_CVPR\_2020\_paper.pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mohamed_Social-STGCNN_A_Social_Spatio-Temporal_Graph_Convolutional_Neural_Network_for_Human_CVPR_2020_paper.pdf)  
24. opendilab/awesome-driving-behavior-prediction \- GitHub, 12월 14, 2025에 액세스, [https://github.com/opendilab/awesome-driving-behavior-prediction](https://github.com/opendilab/awesome-driving-behavior-prediction)  
25. Tutorial — NetworkX 3.6.1 documentation, 12월 14, 2025에 액세스, [https://networkx.org/documentation/stable/tutorial.html](https://networkx.org/documentation/stable/tutorial.html)  
26. Vehicle Trajectory Prediction Using Hierarchical Graph Neural Network for Considering Interaction among Multimodal Maneuvers \- MDPI, 12월 14, 2025에 액세스, [https://www.mdpi.com/1424-8220/21/16/5354](https://www.mdpi.com/1424-8220/21/16/5354)  
27. GL-STGCNN: Enhancing Multi-Ship Trajectory Prediction with MPC Correction \- MDPI, 12월 14, 2025에 액세스, [https://www.mdpi.com/2077-1312/12/6/882](https://www.mdpi.com/2077-1312/12/6/882)  
28. Traffic Agents Trajectory Prediction Based on Spatial–Temporal Interaction Attention \- PMC, 12월 14, 2025에 액세스, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10534871/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10534871/)  
29. \[2502.16157\] Advanced Text Analytics \-- Graph Neural Network for Fake News Detection in Social Media \- arXiv, 12월 14, 2025에 액세스, [https://arxiv.org/abs/2502.16157](https://arxiv.org/abs/2502.16157)  
30. \[Discussion\] Papers with fake NOVEL APPROACH in ML and DL models \- Reddit, 12월 14, 2025에 액세스, [https://www.reddit.com/r/MachineLearning/comments/1go50wf/discussion\_papers\_with\_fake\_novel\_approach\_in\_ml/](https://www.reddit.com/r/MachineLearning/comments/1go50wf/discussion_papers_with_fake_novel_approach_in_ml/)  
31. Python | Visualize graphs generated in NetworkX using Matplotlib \- GeeksforGeeks, 12월 14, 2025에 액세스, [https://www.geeksforgeeks.org/python/python-visualize-graphs-generated-in-networkx-using-matplotlib/](https://www.geeksforgeeks.org/python/python-visualize-graphs-generated-in-networkx-using-matplotlib/)  
32. Graph Visualization: 7 Steps from Easy to Advanced | Towards Data Science, 12월 14, 2025에 액세스, [https://towardsdatascience.com/graph-visualization-7-steps-from-easy-to-advanced-4f5d24e18056/](https://towardsdatascience.com/graph-visualization-7-steps-from-easy-to-advanced-4f5d24e18056/)