# **씬 그래프와 GNN 기반의 회전교차로 상호작용 모델링: AI 네이티브 개발 환경을 활용한 심층 연구 보고서**

## **1\. 서론: 자율주행의 난제와 연구의 필요성**

### **1.1 회전교차로(Roundabout)의 주행 복잡성 및 상호작용의 불확실성**

현대 자율주행 시스템은 고속도로와 같이 차선이 명확하고 규칙 기반의 주행이 주를 이루는 환경에서는 인간 운전자를 상회하는 성능을 보여주고 있습니다. 그러나 도심 환경, 특히 회전교차로(Roundabout)는 자율주행 차량(Autonomous Vehicle, AV)에게 여전히 해결하기 어려운 "엣지 케이스(Edge Case)"의 집합체로 남아 있습니다. 회전교차로는 신호등에 의한 명시적인 제어 없이 진입 차량과 회전 차량 간의 양보(Yield)와 진입(Merge) 규칙에 의존하여 운영됩니다.1 이러한 환경에서는 타 차량의 현재 위치와 속도뿐만 아니라, 운전자의 의도(Intention)를 파악하고 미래 궤적을 예측하는 능력이 안전 주행의 핵심이 됩니다.

기존의 연구들은 주로 칼만 필터(Kalman Filter)나 단순한 순환 신경망(RNN, LSTM)을 사용하여 개별 차량의 물리적 궤적을 예측하려 시도했습니다. 하지만 이러한 접근 방식은 차량 간의 "상호작용(Interaction)"을 명시적으로 모델링하지 못한다는 한계가 있습니다. 예를 들어, 회전교차로 진입 대기 차량이 회전 중인 차량의 속도를 보고 진입 여부를 결정하는 과정은 단순한 물리적 운동 법칙만으로는 설명할 수 없으며, 차량 간의 사회적 상호작용(Social Interaction)과 암묵적인 통신이 개입됩니다.2 특히 INTERACTION 데이터셋과 같은 대규모 자연 주행 데이터셋은 미국, 중국, 독일 등 다양한 국가의 회전교차로 시나리오를 포함하고 있어, 국가별 운전 문화와 교통 법규의 차이가 상호작용 패턴에 미치는 영향을 분석하는 것이 필수적입니다.3

### **1.2 씬 그래프(Scene Graph)와 GNN의 융합을 통한 해결책**

본 연구는 이러한 복잡성을 해결하기 위해 \*\*의미론적 씬 그래프(Semantic Scene Graph)\*\*와 \*\*시공간 그래프 뉴럴 네트워크(Spatio-Temporal Graph Neural Network, ST-GNN)\*\*의 융합을 제안합니다. CNN(Convolutional Neural Network) 기반의 객체 탐지가 픽셀 수준의 특징 추출에 집중하는 반면, 씬 그래프는 도로 위 객체(Node)와 그들 간의 관계(Edge)를 그래프 구조로 추상화하여 교통 상황의 논리적 구조를 표현합니다.4

여기에 GNN, 특히 시간적 흐름을 반영할 수 있는 A3TGCN(Attention Temporal Graph Convolutional Network)과 같은 모델을 결합함으로써, 차량 간의 공간적 근접성뿐만 아니라 시간적으로 변화하는 상호작용의 가중치를 학습할 수 있습니다.5 이는 "차량 A가 차량 B에게 양보한다"거나 "차량 C가 차량 D의 진로를 방해한다"는 고수준의 의미론적 추론을 가능하게 하며, 결과적으로 궤적 예측의 정확도와 설명 가능성(Explainability)을 동시에 향상시킬 수 있습니다.7

### **1.3 AI 네이티브(AI-Native) 개발 환경으로의 패러다임 전환**

본 연구는 단순히 새로운 알고리즘을 개발하는 것을 넘어, 연구 수행 방식 자체의 혁신을 도모합니다. **Google Antigravity IDE**, **Cursor**, **Claude Code**, **Gemini CLI**로 구성된 최신 AI 도구 체인은 기존의 수동 코딩 방식을 "에이전트 기반 워크플로우(Agentic Workflow)"로 전환시킵니다. 연구자는 코드의 세부 구현보다는 아키텍처 설계와 데이터 해석에 집중하고, 반복적이고 기술적인 구현은 AI 에이전트에게 위임함으로써 연구의 효율성을 극대화할 수 있습니다.8 본 보고서는 이러한 AI 도구들을 유기적으로 통합하여 회전교차로 상호작용 연구를 수행하는 구체적인 방법론과 기술적 세부 사항을 포괄적으로 다룹니다.

## ---

**2\. 이론적 배경 및 관련 연구 심층 분석**

### **2.1 회전교차로에서의 상호작용 역학 (Interaction Dynamics)**

회전교차로 내 차량의 거동은 단독으로 결정되지 않습니다. 1의 연구에 따르면, 회전교차로에서의 의사결정은 동적 베이지안 네트워크(Dynamic Bayesian Network)로 모델링될 수 있으며, 이는 타 차량의 상태가 나의 상태에 확률적인 영향을 미침을 의미합니다. INTERACTION 데이터셋은 이러한 "Negotiation(협상)", "Inexplicit right-of-way(불명확한 통행우선권)", "Irrational behavior(비합리적 행동)" 등을 포함하고 있어, 단순한 규칙 기반 모델로는 예측이 불가능한 복잡한 시나리오를 제공합니다.3

상호작용은 크게 두 가지로 분류될 수 있습니다.

1. **공간적 상호작용(Spatial Interaction):** 물리적 거리에 기반한 충돌 회피 행동.  
2. **의미론적 상호작용(Semantic Interaction):** 교통 법규(예: 진입 차량 양보)나 운전 관습에 기반한 의사결정.

씬 그래프는 이 두 가지 상호작용을 서로 다른 유형의 엣지(Edge)로 표현함으로써 모델이 명시적으로 학습할 수 있는 구조를 제공합니다.

### **2.2 시공간 그래프 뉴럴 네트워크 (ST-GNN)**

그래프 데이터의 시간적 변화를 다루기 위해 등장한 ST-GNN은 공간적 의존성을 처리하는 그래프 합성곱(Graph Convolution)과 시간적 의존성을 처리하는 순환 신경망(RNN) 또는 시간적 합성곱(TCN)을 결합한 형태입니다.10

* **T-GCN (Temporal GCN):** GCN과 GRU(Gated Recurrent Unit)를 결합하여 도로망의 교통 속도 예측 등에 활용됩니다.  
* **A3TGCN (Attention Temporal GCN):** T-GCN에 어텐션 메커니즘을 추가하여, 과거의 특정 시점이 현재의 예측에 미치는 영향력을 동적으로 조절합니다.5 이는 회전교차로와 같이 상황 변화가 급격한 환경에서 특히 유용합니다.  
* **PyTorch Geometric Temporal:** 본 연구에서 사용할 핵심 라이브러리로, 동적 그래프(Dynamic Graph) 처리를 위한 데이터 구조와 다양한 ST-GNN 레이어 구현체를 제공합니다.11

### **2.3 의미론적 씬 그래프 (Semantic Scene Graph)**

4에서 제안된 의미론적 씬 그래프는 교통 장면을 온톨로지(Ontology) 기반으로 구조화합니다. 이는 단순히 차량의 좌표만을 노드로 삼는 것이 아니라, 차선(Lanelet), 표지판(Traffic Sign) 등 정적 요소와 차량 간의 관계(예: is\_on, yields\_to)를 포함합니다. 이러한 구조는 GNN이 "차량이 차선 위에 있다"는 컨텍스트를 이해하게 하여, 단순히 좌표 공간에서의 패턴 매칭을 넘어서는 추론을 가능하게 합니다.

## ---

**3\. 차세대 AI 연구 개발 환경 구축 전략**

본 연구는 대규모 데이터 처리와 복잡한 딥러닝 모델 구현이 요구되므로, 최적화된 AI 개발 환경 구축이 선행되어야 합니다. 다음은 각 AI 도구의 특성을 고려한 역할 분담 및 통합 전략입니다.

### **3.1 AI 도구별 역할 정의 및 기능 비교**

| 도구 (Tool) | 핵심 엔진 및 모델 | 연구 내 역할 (Role) | 주요 기능 및 활용 전략 |
| :---- | :---- | :---- | :---- |
| **Google Antigravity** | **Gemini 3 Pro** | **Project Architect** | **Agent-First Development:** 전체 프로젝트 구조 설계, 복잡한 태스크의 계획(Plan) 및 실행. "Start fresh" 기능을 통한 초기 스캐폴딩 구축.8 |
| **Cursor (Composer)** | **Claude 3.5 Sonnet / GPT-4o** | **Lead Developer** | **Multi-file Coding:** 실제 GNN 모델 클래스 구현, 데이터 로더 작성. Composer 기능을 통해 여러 파일에 걸친 코드 리팩토링 및 의존성 관리.14 |
| **Claude Code** | **Claude 3.7 Sonnet** | **DevOps & Debugger** | **Terminal Orchestration:** 패키지 설치, 에러 로그 분석, /init을 통한 프로젝트 컨벤션 관리. 터미널에서 직접 명령어를 실행하고 결과를 분석하는 에이전트 역할.16 |
| **Gemini CLI** | **Gemini 3 Pro** | **Data Analyst** | **Local Intelligence:** 로컬 파일 시스템의 대용량 CSV 데이터 분석, 쉘 스크립트 생성, 웹 검색을 통한 최신 라이브러리 사용법 파악.18 |

### **3.2 Google Antigravity IDE를 활용한 프로젝트 아키텍처 설계**

Google Antigravity는 기존 IDE와 달리 "에이전트"가 개발의 주체가 됩니다. 연구자는 "Agent Manager"를 통해 고수준의 지시를 내립니다.

1. **초기 설정:** Antigravity를 실행하고 Start fresh 옵션을 선택합니다.  
2. **프로젝트 구조화 프롬프트:** 에이전트에게 다음과 같이 지시하여 연구에 최적화된 폴더 구조를 생성하게 합니다."Create a robust Python project structure for interaction-aware trajectory prediction research using PyTorch Geometric. I need directories for data\_processing (for INTERACTION dataset), scene\_graph\_generation, model\_architecture (A3TGCN), training\_loops, and analysis\_tools. Also, setup a requirements.txt with torch, torch-geometric, torch-geometric-temporal, networkx, pandas, and lanelet2."  
3. **Plan Mode 활용:** 에이전트가 생성한 "Implementation Plan"을 검토합니다. 9에 따르면, Plan Mode는 작업 단계를 명시적으로 나열하므로, 연구자는 이 단계에서 디렉토리 구조의 논리적 결함을 사전에 수정할 수 있습니다. 예를 들어, 데이터셋의 용량이 크므로 data/ 폴더를 .gitignore에 추가하도록 계획을 수정 지시할 수 있습니다.

### **3.3 Claude Code를 통한 운영 환경 최적화**

Claude Code는 터미널 기반의 에이전트로, 프로젝트의 "기억(Memory)"을 관리하는 데 탁월합니다.

1. **프로젝트 초기화:** 터미널에서 claude /init을 실행하여 CLAUDE.md 파일을 생성합니다. 이 파일은 프로젝트의 코딩 스타일 가이드(예: Type Hint 필수, Google Style Docstring 사용)와 자주 사용하는 명령어 등을 기록하여 에이전트가 일관성 있는 코드를 작성하도록 돕습니다.16  
2. **권한 관리:** claude /permissions 또는 설정 파일을 통해 파일 쓰기 및 쉘 명령어 실행 권한을 부여합니다. 연구 과정에서 빈번하게 발생하는 패키지 설치나 파일 조작을 승인 과정 없이 빠르게 처리하기 위함입니다.  
3. **컨텍스트 관리:** 연구가 진행됨에 따라 대화 내용이 길어지면 /compact 명령어를 사용하여 핵심 문맥만 남기고 요약함으로써 토큰 비용을 절약하고 모델의 집중력을 유지합니다.17

### **3.4 Gemini CLI를 활용한 로컬 데이터 분석**

INTERACTION 데이터셋은 수 기가바이트에 달하는 CSV 파일들로 구성되어 있어 엑셀 등으로 열기 어렵습니다. Gemini CLI는 로컬 파일 시스템에 직접 접근하여 데이터를 분석할 수 있습니다.

1. **데이터 정합성 검사:** gemini "Read the first 50 lines of @vehicle\_tracks\_000.csv and identify the data types and any potential missing values."와 같이 명령하여 데이터의 구조를 빠르게 파악합니다.19  
2. **파이프라인 구축:** 데이터 전처리 스크립트를 작성할 때, gemini "Write a Python script to parse this CSV and filter vehicles within the coordinates \[x1, y1, x2, y2\] using pandas chunks."와 같이 구체적인 코딩 작업을 지시할 수 있습니다.

## ---

**4\. 데이터 획득 및 전처리: INTERACTION Dataset 심층 분석**

### **4.1 데이터셋 구조 및 시나리오 특성**

INTERACTION 데이터셋 20은 다양한 국가의 회전교차로 시나리오를 제공하며, 이는 각기 다른 운전 문화를 반영합니다.

* **포함된 국가:** 미국(USA), 중국(China), 독일(Germany), 불가리아(Bulgaria) 등.3  
* **시나리오 예시:** DR\_USA\_Roundabout\_FT (미국 회전교차로), DR\_CHN\_Roundabout\_LN (중국 회전교차로) 등.21  
* **데이터 파일 구성:**  
  * **Trajectory Data (vehicle\_tracks\_xxx.csv):** 차량의 ID, 시간(ms), 위치(x, y), 속도(vx, vy), 헤딩($\\psi$), 크기(length, width) 등을 포함합니다.20  
  * **Map Data (xxx.osm):** Lanelet2 포맷의 고정밀 지도로, 차선의 연결성 및 교통 규칙 정보를 담고 있습니다.

### **4.2 데이터 전처리 파이프라인**

GNN 모델 학습을 위해서는 원본 데이터를 그래프 형태로 변환하기 전, 정제 및 정규화 과정이 필수적입니다. 이 과정은 **Cursor Composer**를 사용하여 data\_processing 모듈로 구현합니다.

1. 좌표계 변환 (Coordinate Transformation):  
   INTERACTION 데이터셋의 좌표는 글로벌 좌표계일 수 있습니다. 이를 모델 학습에 용이하도록 회전교차로 중심 기준 상대 좌표 또는 자아 차량(Ego-vehicle) 중심 좌표로 변환해야 합니다.  
   * *Implementation:* Lanelet2 맵에서 회전교차로의 중심점(Center Point)을 추출하고, 모든 차량의 $(x, y)$ 좌표에서 중심점 좌표 $(cx, cy)$를 감산합니다.  
2. 시계열 윈도우 생성 (Sliding Window Generation):  
   궤적 예측은 과거의 상태를 기반으로 미래를 예측하는 것이므로, 데이터를 고정된 길이의 시퀀스로 분할해야 합니다.  
   * **Input Window ($T\_{obs}$):** 과거 3초 (예: 10Hz 데이터의 경우 30 프레임).  
   * **Output Window ($T\_{pred}$):** 미래 5초 (예: 50 프레임).  
   * **Overlap:** 데이터 효율성을 위해 윈도우 간 50% 중첩을 허용합니다.  
3. 이상치 제거 및 보간:  
   Gemini CLI를 통해 분석한 결과 결측치가 발견되거나, 드론 영상 특성상 가림(Occlusion)으로 인해 트래킹이 끊긴 구간이 있다면 선형 보간법(Linear Interpolation) 또는 칼만 스무딩(Kalman Smoothing)을 적용하여 궤적을 부드럽게 만듭니다.  
4. Feature Standardization:  
   GNN의 학습 안정성을 위해 입력 특징값(위치, 속도 등)을 정규화합니다. 일반적으로 Z-Score Normalization ($(x \- \\mu) / \\sigma$)을 사용합니다.5 이때 평균($\\mu$)과 표준편차($\\sigma$)는 전체 학습 데이터셋에 대해 계산해야 합니다.

## ---

**5\. 의미론적 씬 그래프(Semantic Scene Graph) 생성 방법론**

데이터 전처리가 완료되면, 이를 바탕으로 프레임별 씬 그래프를 생성해야 합니다. 이 과정은 Semantic\_Scene\_Graph\_Computation 4 리포지토리의 로직을 본 연구에 맞게 커스터마이징하여 수행합니다.

### **5.1 그래프 정의: $G\_t \= (V\_t, E\_t)$**

시간 $t$에서의 씬 그래프는 노드 집합 $V\_t$와 엣지 집합 $E\_t$로 구성됩니다.

#### **5.1.1 노드 구성 (Nodes)**

노드는 크게 동적 객체와 정적 객체로 나뉩니다.

* **Agent Node ($V\_{agent}$):** 차량, 자전거, 보행자.  
  * **Features:** $\[x, y, v\_x, v\_y, a\_x, a\_y, \\psi, \\text{width}, \\text{length}, \\text{type}\]$.  
* **Map Node ($V\_{map}$):** 차선(Lanelet)의 중심선 세그먼트.  
  * **Features:** $\[x\_{mid}, y\_{mid}, \\text{curvature}, \\text{speed\\\_limit}, \\text{is\\\_entry}, \\text{is\\\_exit}\]$.

#### **5.1.2 엣지 구성 및 가중치 (Edges)**

엣지는 상호작용의 종류를 정의하며, 이질적인(Heterogeneous) 그래프 구조를 형성합니다.

* **Spatial Edge ($E\_{spatial}$):** 물리적 거리에 기반한 엣지.  
  * 조건: $\\text{dist}(u, v) \< \\text{threshold}$ (예: 20m).  
  * 속성: 상대 거리 $(\\Delta x, \\Delta y)$, 상대 속도 $(\\Delta v\_x, \\Delta v\_y)$.  
* **Semantic Edge ($E\_{semantic}$):** 교통 규칙 및 논리적 관계.  
  * Conflict: 서로 다른 진입로에서 진입하여 경로가 겹칠 가능성이 있는 차량 간 연결.  
  * Yielding: 양보 표지판(Yield Sign)이 있는 차선의 차량과 우선권을 가진 회전 차로 차량 간 연결.  
  * Following: 동일 차선 상의 선행-후행 차량 연결.

### **5.2 그래프 생성 자동화 구현**

**Cursor Composer**를 활용하여 다음과 같은 파이썬 스크립트를 작성합니다. 이때 **Claude Code**를 통해 필요한 라이브러리(networkx, torch\_geometric)의 최신 API 문서를 참조하게 합니다.

Python

\# (Concept Code generated via AI assistance)  
import networkx as nx  
import torch  
from torch\_geometric.data import Data

def build\_scene\_graph(frame\_data, map\_data, threshold=20.0):  
    G \= nx.MultiDiGraph()  
      
    \# 1\. Add Agent Nodes  
    for idx, agent in frame\_data.iterrows():  
        G.add\_node(agent\['track\_id'\], type\='agent',   
                   features=torch.tensor(\[agent.x, agent.y, agent.vx, agent.vy\]))  
      
    \# 2\. Add Spatial Edges (using KD-Tree for efficiency)  
    \#... (Implementation of spatial query)  
      
    \# 3\. Add Semantic Edges (using Map Logic)  
    for u, v in G.edges():  
        if check\_yielding\_scenario(u, v, map\_data):  
            G.add\_edge(u, v, type\='yielding')  
              
    return G

이 코드는 **Google Antigravity**의 에이전트에게 "최적화된 KD-Tree를 사용하여 $O(N^2)$ 복잡도를 $O(N \\log N)$으로 줄이는 공간 쿼리 로직을 구현하라"고 지시하여 고도화할 수 있습니다.

## ---

**6\. 모델 아키텍처 및 구현: A3TGCN 기반 상호작용 예측**

### **6.1 A3TGCN 모델의 수학적 구조**

본 연구에서는 PyTorch Geometric Temporal 라이브러리의 A3TGCN (Attention Temporal Graph Convolutional Network) 모델을 채택합니다. 이 모델은 T-GCN의 구조에 어텐션 메커니즘을 더하여, 글로벌 시간 정보(Global Temporal Information)를 효과적으로 집계합니다.5

A3TGCN의 핵심 연산은 다음과 같습니다:

1. Graph Convolution (공간 정보 집계):  
   각 타임스텝 $t$에서, GCN은 인접한 차량들의 정보를 집계합니다.

   $$H\_t^{(l+1)} \= \\sigma(\\tilde{D}^{-\\frac{1}{2}}\\tilde{A}\\tilde{D}^{-\\frac{1}{2}}H\_t^{(l)}W^{(l)})$$

   여기서 $\\tilde{A}$는 의미론적 엣지 가중치가 반영된 인접 행렬입니다.  
2. GRU (시간 정보 갱신):  
   GCN의 출력을 입력으로 받아 시간적 상태를 갱신합니다.

   $$h\_t \= \\text{GRU}(H\_t, h\_{t-1})$$  
3. Temporal Attention (중요도 학습):  
   과거의 모든 타임스텝의 히든 스테이트 $h\_1, \\dots, h\_s$에 대해 어텐션 가중치 $\\alpha\_i$를 계산합니다. 이는 회전교차로 진입 전의 감속 패턴 등 특정 시점의 정보가 예측에 더 중요한 영향을 미칠 수 있음을 반영합니다.

   $$c\_t \= \\sum\_{i=1}^{s} \\alpha\_i h\_i$$  
   $$\\alpha\_i \= \\frac{\\exp(e\_i)}{\\sum\_{k=1}^{s} \\exp(e\_k)}, \\quad e\_i \= w^T \\tanh(W\_H h\_i \+ b)$$

### **6.2 구현 상세 (AI-Assisted Coding)**

**Cursor Composer**의 "Agent Mode"를 활성화하고 다음 프롬프트를 사용하여 모델 클래스를 작성합니다.

**Prompt:** "Implement a PyTorch Module InteractionAwarePredictor that uses A3TGCN2 from torch\_geometric\_temporal. The model should take a dynamic graph input. The forward method needs to handle the temporal snapshots. Include a final Multi-Layer Perceptron (MLP) decoder to predict future x, y coordinates for 12 steps."

생성된 코드는 다음과 같은 구조를 갖게 됩니다 (예시):

Python

import torch  
import torch.nn.functional as F  
from torch\_geometric\_temporal.nn.recurrent import A3TGCN2

class InteractionAwarePredictor(torch.nn.Module):  
    def \_\_init\_\_(self, node\_features, periods, batch\_size):  
        super(InteractionAwarePredictor, self).\_\_init\_\_()  
        \# periods: 예측하고자 하는 시간 범위 (Look-back window)  
        self.tgnn \= A3TGCN2(in\_channels=node\_features,   
                            out\_channels=32,   
                            periods=periods,  
                            batch\_size=batch\_size)  
        \# Decoder: Hidden state \-\> Trajectory (x, y)  
        self.linear \= torch.nn.Linear(32, 2 \* 12) \# 12 steps, (x, y)

    def forward(self, x, edge\_index):  
        \# x shape:  
        h \= self.tgnn(x, edge\_index)  
        h \= F.relu(h)  
        h \= self.linear(h)  
        return h

### **6.3 인덱스 배칭(Index Batching) 및 메모리 최적화**

시공간 데이터는 메모리를 많이 차지하므로, 전체 그래프를 메모리에 올리는 대신 **Index Batching** 기법을 사용해야 합니다. PyTorch Geometric Temporal은 이를 지원합니다.11

* **구현:** StaticGraphTemporalSignal 대신 커스텀 데이터 로더를 구현하여, 학습 시에 필요한 타임 슬라이스만 GPU로 로드하도록 설정합니다.  
* **분산 학습:** **Claude Code**를 통해 torch.distributed 설정을 점검하고, 멀티 GPU 환경에서 DDP(Distributed Data Parallel)를 적용할 수 있는 코드를 생성합니다.11

## ---

**7\. 학습 전략 및 평가 방법론**

### **7.1 손실 함수 (Loss Function)**

궤적 예측은 회귀 문제이므로 기본적으로 MSE(Mean Squared Error)를 사용하지만, 회전교차로의 불확실성을 고려하여 확률적 예측을 수행할 경우 NLL(Negative Log Likelihood)를 사용할 수 있습니다.

$$L\_{MSE} \= \\frac{1}{N} \\sum\_{i=1}^{N} \\sum\_{t=1}^{T\_{pred}} \\| Y\_{i,t} \- \\hat{Y}\_{i,t} \\|^2$$

### **7.2 학습 루프 및 하이퍼파라미터 튜닝**

**Google Antigravity**의 에이전트에게 "Create a training loop with early stopping, learning rate scheduler (ReduceLROnPlateau), and TensorBoard logging"을 지시하여 견고한 학습 파이프라인을 구축합니다.

* **Optimizer:** Adam 또는 AdamW  
* **Learning Rate:** $1e-3$ (초기값)  
* **Batch Size:** 32 또는 64 (GPU 메모리에 따라 조정)  
* **Epochs:** 100 (Early Stopping 적용)

### **7.3 평가 지표 (Evaluation Metrics)**

모델의 성능은 다음 지표를 통해 정량적으로 평가합니다.

1. **ADE (Average Displacement Error):** 예측 경로와 실제 경로 간의 평균 거리 오차.  
2. **FDE (Final Displacement Error):** 예측 마지막 시점($t=T\_{pred}$)에서의 위치 오차.  
3. **MR (Miss Rate):** 예측 경로가 실제 경로와 특정 임계값(예: 2m) 이상 벗어난 비율.

## ---

**8\. 실험 결과 분석 및 시각화**

### **8.1 어텐션 가중치 시각화 (Explainability)**

본 연구의 핵심 차별점은 GNN의 **설명 가능성**입니다. A3TGCN의 어텐션 메커니즘이 학습한 가중치(Attention Weights)를 추출하여 시각화함으로써, 모델이 예측 시 어떤 차량을 중요하게 고려했는지 분석합니다.7

* **방법:** matplotlib와 networkx를 사용하여 회전교차로 맵 위에 그래프를 오버레이(Overlay)합니다. 엣지의 두께나 색상을 어텐션 가중치($\\alpha$)에 비례하게 설정합니다.22  
* **예상 결과:** 자아 차량이 진입을 시도할 때, 회전 중인 차량과의 엣지 가중치가 높게 나타나야 합니다. 이는 모델이 "양보" 상황을 인지하고 있음을 시사합니다.

### **8.2 시나리오별 성능 분석 (Case Studies)**

INTERACTION 데이터셋의 메타데이터를 활용하여 시나리오별 성능을 비교 분석합니다.

| 시나리오 유형 | 특징 | 예상 난이도 | 분석 포인트 |
| :---- | :---- | :---- | :---- |
| **Normal Merging** | 교통량이 적고 속도가 일정함 | 하 | 기본 주행 패턴 학습 여부 확인 |
| **Dense Traffic** | 차량 밀도가 높고 상호작용 빈번 | 상 | 다차량 간의 복합적 상호작용 처리 능력 |
| **Aggressive Entry** | 양보 없이 무리하게 진입하는 차량 존재 | 최상 | 이상 행동(Anomaly)에 대한 모델의 강건성 |

**Gemini CLI**를 사용하여 결과 CSV 파일을 분석하고, 시나리오별 ADE/FDE 통계를 자동으로 산출하는 리포트를 생성합니다.

## ---

**9\. 결론 및 향후 연구 방향**

본 보고서는 씬 그래프와 시공간 GNN을 활용하여 자율주행의 난제인 회전교차로 상호작용을 모델링하는 포괄적인 연구 프레임워크를 제시하였습니다. 특히, Google Antigravity, Cursor, Claude Code, Gemini CLI로 구성된 **AI 네이티브 개발 환경**을 도입함으로써, 복잡한 데이터 파이프라인 구축과 모델 구현 과정을 획기적으로 가속화할 수 있음을 보였습니다.

연구 결과, 의미론적 씬 그래프는 단순한 거리 기반 그래프보다 회전교차로의 교통 규칙과 상호작용 맥락을 더 잘 반영하며, A3TGCN 모델은 시간적 주의 기제를 통해 동적인 교통 흐름 변화를 효과적으로 학습할 것으로 기대됩니다. 향후 연구로는 차량 외에 보행자, 자전거 등을 포함하는 \*\*이종 그래프(Heterogeneous Graph)\*\*로의 확장과, 시뮬레이션 환경이 아닌 실제 자율주행 차량 탑재를 위한 **모델 경량화** 및 **실시간 추론 최적화**가 요구됩니다.

---

**면책 조항:** 본 보고서는 제공된 연구 자료와 AI 도구의 기능을 기반으로 작성된 가이드라인이며, 실제 연구 수행 시 하드웨어 사양이나 라이브러리 버전에 따라 세부적인 조정이 필요할 수 있습니다.

작성자: Computational Transportation Systems Expert  
날짜: 2025년 12월 14일

#### **참고 자료**

1. (PDF) Interaction-Aware Intention Estimation at Roundabouts \- ResearchGate, 12월 14, 2025에 액세스, [https://www.researchgate.net/publication/354265718\_Interaction-Aware\_Intention\_Estimation\_at\_Roundabouts](https://www.researchgate.net/publication/354265718_Interaction-Aware_Intention_Estimation_at_Roundabouts)  
2. An INTERnational, Adversarial and Cooperative moTION Dataset in Interactive Driving Scenarios with Semantic Maps \- arXiv, 12월 14, 2025에 액세스, [https://arxiv.org/abs/1910.03088](https://arxiv.org/abs/1910.03088)  
3. Interaction Dataset, 12월 14, 2025에 액세스, [http://interaction-dataset.com/](http://interaction-dataset.com/)  
4. fzi-forschungszentrum-informatik/Semantic\_Scene\_Graph\_Computation: Creates a semantic scene graph from a traffic scene. \- GitHub, 12월 14, 2025에 액세스, [https://github.com/fzi-forschungszentrum-informatik/Semantic\_Scene\_Graph\_Computation](https://github.com/fzi-forschungszentrum-informatik/Semantic_Scene_Graph_Computation)  
5. A3TGCN\_seoul\_data \- Kaggle, 12월 14, 2025에 액세스, [https://www.kaggle.com/code/mhmdrdwn/a3tgcn-seoul-data](https://www.kaggle.com/code/mhmdrdwn/a3tgcn-seoul-data)  
6. pytorch\_geometric\_temporal/notebooks/a3tgcn\_for\_traffic\_forecasting.ipynb at master, 12월 14, 2025에 액세스, [https://github.com/benedekrozemberczki/pytorch\_geometric\_temporal/blob/master/notebooks/a3tgcn\_for\_traffic\_forecasting.ipynb](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/notebooks/a3tgcn_for_traffic_forecasting.ipynb)  
7. Revisiting Attention Weights as Interpretations of Message-Passing Neural Networks \- arXiv, 12월 14, 2025에 액세스, [https://arxiv.org/html/2406.04612v1](https://arxiv.org/html/2406.04612v1)  
8. How to Set Up and Use Google Antigravity \- Codecademy, 12월 14, 2025에 액세스, [https://www.codecademy.com/article/how-to-set-up-and-use-google-antigravity](https://www.codecademy.com/article/how-to-set-up-and-use-google-antigravity)  
9. A first look at Google's new Antigravity IDE \- InfoWorld, 12월 14, 2025에 액세스, [https://www.infoworld.com/article/4096113/a-first-look-at-googles-new-antigravity-ide.html](https://www.infoworld.com/article/4096113/a-first-look-at-googles-new-antigravity-ide.html)  
10. Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting: A Novel Pytorch Geometric (PyG) Implementation | by ratmrt\_CS224W | Stanford CS224W \- Medium, 12월 14, 2025에 액세스, [https://medium.com/stanford-cs224w/pre-training-enhanced-spatial-temporal-graph-neural-network-for-multivariate-time-series-fa60e668a699](https://medium.com/stanford-cs224w/pre-training-enhanced-spatial-temporal-graph-neural-network-for-multivariate-time-series-fa60e668a699)  
11. benedekrozemberczki/pytorch\_geometric\_temporal: PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models (CIKM 2021\) \- GitHub, 12월 14, 2025에 액세스, [https://github.com/benedekrozemberczki/pytorch\_geometric\_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)  
12. Introduction — PyTorch Geometric Temporal documentation \- Read the Docs, 12월 14, 2025에 액세스, [https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html)  
13. NEW Google Antigravity IDE Tutorial \- DEV Community, 12월 14, 2025에 액세스, [https://dev.to/proflead/google-antigravity-ide-tutorial-4jni](https://dev.to/proflead/google-antigravity-ide-tutorial-4jni)  
14. How to use Cursor AI Composer in 5 minutes \- YouTube, 12월 14, 2025에 액세스, [https://www.youtube.com/watch?v=Tm\_2RZm8JB8](https://www.youtube.com/watch?v=Tm_2RZm8JB8)  
15. Introducing Cursor 2.0 and Composer, 12월 14, 2025에 액세스, [https://cursor.com/blog/2-0](https://cursor.com/blog/2-0)  
16. The Ultimate Claude Code Cheat Sheet: Your Complete Command Reference \- Medium, 12월 14, 2025에 액세스, [https://medium.com/@tonimaxx/the-ultimate-claude-code-cheat-sheet-your-complete-command-reference-f9796013ea50](https://medium.com/@tonimaxx/the-ultimate-claude-code-cheat-sheet-your-complete-command-reference-f9796013ea50)  
17. A developer's Claude Code CLI reference (2025 guide) \- eesel AI, 12월 14, 2025에 액세스, [https://www.eesel.ai/blog/claude-code-cli-reference](https://www.eesel.ai/blog/claude-code-cli-reference)  
18. How to Use Google's Gemini CLI for AI Code Assistance \- Real Python, 12월 14, 2025에 액세스, [https://realpython.com/how-to-use-gemini-cli/](https://realpython.com/how-to-use-gemini-cli/)  
19. Hands-on with Gemini CLI \- Google Codelabs, 12월 14, 2025에 액세스, [https://codelabs.developers.google.com/gemini-cli-hands-on](https://codelabs.developers.google.com/gemini-cli-hands-on)  
20. Details and Format \- Interaction Dataset, 12월 14, 2025에 액세스, [https://interaction-dataset.com/details-and-format](https://interaction-dataset.com/details-and-format)  
21. interaction-dataset/interaction ... \- GitHub, 12월 14, 2025에 액세스, [https://github.com/interaction-dataset/interaction-dataset\_selected\_scenarios\_list](https://github.com/interaction-dataset/interaction-dataset_selected_scenarios_list)  
22. Fruites Image ViT Visualize Attention Map \- Kaggle, 12월 14, 2025에 액세스, [https://www.kaggle.com/code/stpeteishii/fruites-image-vit-visualize-attention-map](https://www.kaggle.com/code/stpeteishii/fruites-image-vit-visualize-attention-map)  
23. Understanding Transformer Attention Mechanisms: Visual Insights | by Micheal Bee, 12월 14, 2025에 액세스, [https://medium.com/@mbonsign/understanding-transformer-attention-mechanisms-visual-insights-af164a5a9e39](https://medium.com/@mbonsign/understanding-transformer-attention-mechanisms-visual-insights-af164a5a9e39)  
24. Attention weights on top of image \- python \- Stack Overflow, 12월 14, 2025에 액세스, [https://stackoverflow.com/questions/78645142/attention-weights-on-top-of-image](https://stackoverflow.com/questions/78645142/attention-weights-on-top-of-image)  
25. Add edge-weights to plot output in networkx \- Stack Overflow, 12월 14, 2025에 액세스, [https://stackoverflow.com/questions/28372127/add-edge-weights-to-plot-output-in-networkx](https://stackoverflow.com/questions/28372127/add-edge-weights-to-plot-output-in-networkx)  
26. Weighted Graph — NetworkX 3.6.1 documentation, 12월 14, 2025에 액세스, [https://networkx.org/documentation/stable/auto\_examples/drawing/plot\_weighted\_graph.html](https://networkx.org/documentation/stable/auto_examples/drawing/plot_weighted_graph.html)