# Multimodal Learning

### Multimodal Learning with Transformers: A Survey

## Abstract

- multimodal data: 텍스트, 이미지, 비디오, 오디오와 같이 여러 형태의 데이터를 말함
- 이러한 데이터를 처리하는 트랜스포머 모델이 최근 딥러닝 분야에서 주목받고 있음
- 주요 내용
    - 멀티모달 학습의 배경: 멀티모달 학습이 왜 중요한지, 그리고 어떤 방식으로 다양한 데이터 타입을 통합하는지
    - 트랜스포머 기술의 생태계: 트랜스포머 모델이 어떻게 발전했으며, 현재 어떤 다양한 변형과 응용이 있는지
    - 멀티모달 빅데이터 시대: 대규모 데이터셋을 처리하는 멀티모달 트랜스포머의 역할과 중요성
    - 기하학적 및 위상학적 관점에서의 분석: 기존의 트랜스포머 모델들과 멀티모달 변형들이 어떻게 구조적으로 다른지를 기하학적이나 위상학적 관점에서 체계적으로 검토
        - 기하학적 관점: 데이터나 모델의 구조를 공간적인 형태로 해석. 예를 들어, 고차원 데이터를 저차원으로 투영하거나 시각화하여 데이터 간의 관계를 이해하려고 할 때 기하학적 개념이 활용됨. 딥러닝 모델의 학습 과정을 기하학적 경로로 해석하여 모델이 어떻게 최적의 해를 찾아가는지 분석할 수 있음
        - 위상학적 관점: 위상학은 공간의 연결성과 연속성을 다루는 수학의 한 분야로, 형태가 변해도 유지되는 속성을 연구함. 위상학적 데이터 분석은 데이터의 모양이나 구조를 중심으로 연구하며, 이를 통해 데이터 사이의 근본적인 관계나 패턴을 파악할 수 있음. 예를 들어, 복잡한 데이터 세트에서 중요한 클러스터나 구멍(데이터가 없는 공간) 등을 파악하는데 사용될 수 있음
    - 멀티모달 pre-training과 application: 멀티모달 트랜스포머가 사전 학습되는 방식과 특정 멀티모달 작업을 수행하는 방법
    - 공통 과제와 설계 요약: 멀티모달 트랜스포머 모델과 application이 직면한 공통적인 도전과 그 설계에 대한 요약
    - 개방형 문제와 연구 방향: 커뮤니티가 직면한 개방형 문제와 향후 연구 방향

## Introduction

- transformer를 사용한 멀티모달 학습에 초점을 맞춤
- transformer는 여러 토큰 시퀀스를 입력으로 받을 수 있으며, 각 시퀀스는 다양한 속성을 포함할 수 있음. 이는 멀티모달 데이터를 자연스럽게 처리할 수 있게 해주며, 아키텍처의 큰 수정 없이도 멀티모달 학습을 수행할 수 있게 함
- transformer의 핵심 기능인 self attention은 멀티모달 데이터 간의 상호 연관성(inter-modal correlation)을 학습하는데 중요함. 이를 통해 각 모달의 특성(per-modal specificity)을 학습하는 것도 간단해짐. 셀프 어텐션은 입력 데이터 간의 관계를 유연하게 모델링할 수 있어 멀티모달 학습에 매우 적합함.
- 이 논문은 트랜스포머 기반의 멀티모달 머신러닝 분야의 현재 상태에 대한 포괄적인 검토를 제공함

## Background

### Multimodal Learning (MML)

- 멀티모달 학습은 여러 가지 모드(예: 텍스트, 이미지, 오디오 등)의 데이터를 이해하고 처리하는 인공지능 및 기계학습 분야를 가리킴
- 초기 multimodal application은 인간 사회의 핵심적인 특성을 모방하려는 시도였음
- 최근 대규모 언어 모델과 이를 기반으로 한 멀티모달 모델의 성공은 트랜스포머 아키텍처의 잠재력을 더욱 확장하고 있음을 보여줌

### Transformers: a Brief History and Milestones

- vanilla transformer
    - 트랜스포머는 시퀀스 데이터의 representation learning을 위한 혁신적인 모델로, 초반에는 self-attention mechanism을 특징으로 했음
    - 다양한 자연어 처리 작업에서 선도적인 결과를 보였음
    - 그 후에는 BERT, BART, GPT, Longformer, transformer-XL, XLNet 등과 같은 파생 모델들이 소개됐음
- visual domain
    - 비전(이미지와 관련된) 분야에서는 일반적으로 CNN 특징과 표준 트랜스포머 인코더를 조합하는 파이프라인을 사용함
    - 이를 통해 원시 이미지를 낮은 해상도로 크기 조정하고 1D 시퀀스(이미지의 공간 정보를 일렬로 나열하는 것)로 변환하여 BERT와 유사한 스타일의 사전 훈련을 수행함
- ViT(Vision Transformer)
    - 트랜스포머의 인코더를 이미지에 직접 적용하여 end-to-end 솔루션을 제공하며 low-level tasks, recognition, detection 등 다양한 컴퓨터 비전 작업에 사용됨
- VideoBERT
    - 최초로 트랜스포머 모델을 멀티모달 데이터에 도입한 사례로, 이후 다양한 트랜스포머 기반 멀티모달 사전 훈련 모델들이 연구됐음
- CLIP
    - 멀티모달 사전 훈련 모델을 활용하여 사전 훈련된 모델을 zero-shot recognition을 수행할 수 있는 retrieval 작업(모델이 주어진 질의나 검색어에 대한 관련 정보를 검색하고 반환하는 작업)으로 변환하는 데 사용됐음

### Multimodal Big Data

- Conceptual Captions, COCO, VQA, Visual Genome 등 많은 대규모 멀티모달 데이터셋이 제안됐음
- 데이터셋의 새로운 트렌드
    - 데이터의 규모가 더 커짐
    - 시각, 텍스트, 오디오 외에도 Pano(360도 시야의 파노라마 이미지)-AVQA(Audio-Visual Question Answering) (파노라마 이미지와 관련된 질문 응답 작업을 위한 데이터)와 같은 더 다양한 데이터 형식이 등장하고 있음
    - 자율주행, 금융 데이터 등 다양한 응용 분야에서 연구되고 있음
    - 밈의 혐오 발언과 같은 추상적인 작업도 제안되고 있음(밈 콘텐츠에서 혐오 발언과 같이 추상적이고 주관적인 개념을 탐지하거나 분류하는 작업)
    - instructional video도 인기를 얻고 있음. 컴퓨터 비전과 자연어 처리 모델은 instructional video의 내용을 이해하고 분석하는데 활용될 수 있으며, 이를 통해 동영상 콘텐츠를 자동으로 요약하거나 관련 정보를 검색하는 등의 작업이 가능해짐. 이러한 모델은 교육 분야와 정보 검색 분야에서 유용하게 활용될 수 있음

## Transformers

- vanilla self-attention(transformer)은 위상 기하학적 공간에서 fully-connected graph로 모델링 할 수 있음(모든 특성이 연관되어 있다는 의미)
- 다른 deep network와 비교하여 transformer는 본질적으로 보다 일반적이고 유연한 모델링 공간을 가지고 있음

### Vanilla Transformer

- vanilla transformer는 transformer 모델의 원조 버전으로, encoder-decoder 구조를 가짐
- 입력 데이터는 여러 transformer layer 또는 block에 스택되며, 각 블록은 multi-head self attention(MHSA) 및 fully connected feed-forward network(FFN)와 같은 두 개의 하위 레이어로 구성됨
- gradient back propagation을 위해 residual connection 및 normalization layer가 사용됨
- pre-normalization과 post-normalization이라는 중요한 문제에 대한 연구가 진행 중
    - 원래 vanilla transformer는 각 MHSA와 FFN sub-layer에 대해 post-normalization을 사용하는데, 수학적 관점에서 고려할 때 pre-normalization이 더 합리적임
    - 이는 projection 전에 normalization이 수행되어야 한다는 행렬 이론의 기본 원리와 유사함

**Input Tokenization**

- transformer는 초기에 기계 번역을 위한 모델로 개발됐으므로 토큰화된 시퀀스를 입력으로 사용함
- 원래 self-attention은 양식에 관계 없이 임의의 입력을 fully-connected graph로 모델링 가능
- vanilla transformer와 variant transformer는 모두 토큰화된 시퀀스를 사용하며 각 토큰은 그래프의 노드로 간주
- position embedding은 position 정보를 유지하기 위해 token embedding에 추가됨
    - vanilla transformer에서는 sin, cos function을 사용해 position embedding 생성
    - 시간적/공간적 정보를 제공하기 위한 일종의 spatial information을 제공하기 위한 implicit coordinate basis of feature space로 이해할 수 있음
    - position embedding은 optional
    - self-attention의 수학적 관점(scaled dot-product attention)에서 고려하면 position embedding 정보가 누락된 경우 words(in text), nodes(in graphs)의 위치에 attention이 변하지 않음. 그래서 대부분의 경우 transformer에는 position embedding이 필요
- input tokenization은 보다 일반적인 접근 방식을 제공하며, 다양한 입력 구성 방법을 지원함

**Self-Attention and Multi-Head Self-Attention**

- Self-Attention (SA)
    - self-attention은 transformer 모델의 핵심 요소 중 하나로, 입력 시퀀스 내의 각 요소(토큰)가 다른 모든 요소에 주의를 기울 수 있도록 하는 메커니즘임
    - self-attention을 통해 입력 데이터는 fully-connected 그래프로 인코딩됨. 이것은 입력 시퀀스의 모든 요소 간의 상호 작용을 나타내며, 이것은 transformer encoder를 “fully-connected GNN(Graph Neural Network) encoder”로 비유하게 함
    - self-attention은 모델이 입력 데이터의 글로벌 관계와 패턴을 이해하고 처리하는 데 도움이 되며, 이로써 transformer 모델은 non-local ability of global perception을 가짐
- masked self-attention (MSA)
    - MSA는 transformer 모델의 디코더 부분에서 사용되는 self-attenion의 변형
    - transformer 디코더는 subsequent position에 attending 하는 것을 방지하기 위해 self-attention을 수정해야 하는데, 이를 위해 MSA가 사용됨
    - 기본적으로 MSA는 transformer model에 addtional knowledge를 주입하는 데 사용됨
- multi-head self-attention (MHSA)
    - MHSA는 여러 개의 self-attention 서브 레이어를 병렬로 쌓을 수 있는 구조를 가짐
    - 각 self-attention 서브 레이어는 입력 데이터의 다른 측면 또는 표현을 학습함. 그런 다음 이러한 서브 레이어의 출력은 projection matrix에 의해 concatenated 되어 하나의 MHSA가 형성됨
    - MHSA는 모델이 입력 데이터의 다양한 표현 하위 공간에 공동으로 주의를 기울이는 데 도움을 주며, 더 강력한 모델을 만들 수 있도록 함

**Feed-Forward Network**

- 출력은 non-linear activation이 있는 연속적인 linear layer로 구성된 position-wise feed-forward network를 통과
- 일부 transformer 문헌에서는 FFN을 Multi-Layer Perceptron(MLP)라고도 함

### Vision Transformer

- ViT는 입력 이미지를 작은 고정 크기의 patch로 나누어야 함
- 각각의 패치는 linearly embedded layer를 통과한 후, position embedding을 추가함
- 이렇게 준비된 각 패치는 standard transformer encoder로 encoding 됨. 이 과정에서 각 패치는 트랜스포머의 입력으로 사용되며, 패치 간의 상호 작용을 학습함.

### Multimodal Transformer

**Multimodal Input**

- Multimodal Transformer는 여러 모달(ex. 텍스트, 이미지, 오디오)의 입력을 처리하기 위한 아키텍처로서, GNN의 한 유형으로 이해될 수 있음
- self-attention과 변형된 아키텍처를 사용하여 데이터를 처리하기 전에 입력 데이터를 토큰화하고 각 토큰을 나타내는 임베딩 공간을 선택함
- 다양한 임베딩을 토큰 방식으로 합칠 수 있는 fusion 방법이 일반적으로 사용됨

**Self-Attention Variants in Multimodal Context**

- Multimodal Transformer에서 다른 모달 간의 상호 작용(fusion, alignment)는 기본적으로 self-attention 및 그 변형을 통해 처리됨
- 여러 self-attention 설계 관점에 따라 다양한 방식으로 cross-modal interactions를 다룸
- 이러한 관점은 early summation, early concatenation, hierarchical attention, cross-attention, cross-attention to concatenation 등이 있음
- early summation
    - 간단하고 효과적인 multimodal interaction
    - 여러 modalities의 token embeddings를 각 token position에서 가중치 합산 후 transformer layer로 전달
    - 다른 모달리티의 임베딩을 결합할 때, 각 모달리티의 토큰 임베딩에 가중치를 할당하고 이들을 합산하는 것
    - 가중치는 각 토큰의 중요성을 나타냄
    - 계산이 복잡하지 않으나 가중치를 수동으로 설정해야 할 수 있음
- early concatenation(all-attention/CoTransformer)
    - 각 모달리티의 token embedding sequence를 concatenate하고, 이를 하나의 큰 sequence로 만들어서 transformer layer에 입력으로 제공하는 것을 중점으로 함
    - 큰 sequence는 다양한 모달리티에서 나온 정보를 보유하며, 각 모달리티의 위치 정보를 유지함
    - 이 방식을 사용하면 모든 모달리티의 정보가 통합되어 모델에 입력되므로, 다양한 모달리티 간의 상호작용을 모델링하기에 유용할 수 있음
    - 그러나 early concatenation 방법은 하나의 큰 시퀀스를 생성하므로 시퀀스가 길어질수록 계산량이 증가할 수 있으며, 모든 모달리티에 대한 정보를 함께 처리해야 하므로 모델의 복잡성이 높아질 수 있음
    - VideoBERT
- hierarchical attention (multi-stream → one-stream)
    - 다양한 모달리티의 입력 정보를 각각 독립적인 트랜스포머 스트림으로 인코딩함. 즉, 이미지에 대한 정보를 처리하는 트랜스포머 스트림과 텍스트에 대한 정보를 처리하는 또 다른 트랜스포머 스트림이 별도로 존재함
    - 이러한 독립적인 스트림의 출력은 상호작용을 위해 다른 트랜스포머로 연결되고 융합됨. 예를 들어, 이미지와 텍스트 정보가 서로 영향을 주고 받을 수 있도록 조정됨
    - 각 모달리티의 정보는 독립적으로 인코딩되지만, 다른 모달리티와의 관계를 고려하여 최종 통합된 표현을 얻을 수 있음
- hierarchical attention (one-stream → multi-stream)
    - concatenated multimodal inputs가 두 개의 개별 트랜스포머 스트림이 뒤따르는 단일 스트림 트랜스포머에 의해 인코딩됨
    - cross-modal interaction과 동시에 uni-modal representation의 독립성 유지
    - InterBERT
- cross-attention
    - Query와 Key를 swap하여 cross-modal attention을 수행함. 각 모달리티에서 어떤 부분에 주의해야 하는지를 서로 교환하게 되므로 다른 모달리티 간의 관계를 모델링할 수 있고, 계산 복잡도가 상대적으로 낮음
    - cross-attention은 다른 모달리티 간의 관계를 고려하지만, 각 모달리티 내부의 self-attention을 수행하지 않음. 따라서 모달리티 간의 상호작용을 모델링하면서 각 모달리티 내부 정보를 유지할 수 있음
    - 예를 들어, 이미지와 텍스트가 함께 제공되는 시각적 질문 응답(VQA) 작업에서, cross-attention을 사용하면 이미지에서 질문 텍스트로 주의를 기울이거나 질문 텍스트에서 이미지로 주의를 기울여 질문에 대한 답변을 생성할 수 있음
    - VilBERT에서 처음 제안됐음
- cross-attention to concatenation
    - cross-attention을 통해 얻은 다른 모달리티 간의 상호작용 정보는 각 모달리티의 원래 정보와 함께 concatenate 됨. 각 모달리티의 정보와 상호작용 정보가 결합되어 새로운 표현이 형성됨
    - 연결된 정보는 또 다른 트랜스포머에 의해 처리됨. 이런 종류의 계층적 cross-modal interaction은 cross-attention의 단점을 보완할 수 있음

![Untitled](Multimodal%20Learning%206d7293d0dd984fd9b2ec0dbbc9908658/Untitled.png)

**Network Architectures**

- multimodal transformer는 다양한 아키텍처로 구성될 수 있으며, 내부 멀티모달 어텐션을 통해 작동함
- 아키텍처의 구조는 입력 데이터의 다양한 특징과 모달 간의 상호 작용을 고려해 설계됨
- single-stream: early summation, early concatenation
- multi-streams: cross-attention
- hybrid-streams: hierarchical attention, cross-attention to concatenation

## Application Scenarios

### Transformer for Multimodal Pretraining

**Task-Agnostic Multimodal Pretraining**

- Vision Language Pretraining(VLP): 이미지와 언어 또는 비디오와 언어와 같은 다양한 모달리티 간의 사전훈련. 모델이 이미지와 텍스트 또는 비디오와 텍스트를 함께 이해하고 처리할 수 있도록 하는 것을 목표로 함
- speech can be used as text: 음성 인식 기술에서 멀티모달 컨텍스트를 활용하여 음성을 텍스트로 변환할 수 있음. VideoBERT와 같은 모델에서 사용되는 방법 중 하나
- well-aligned multimodal data에 지나치게 의존: 현재의 멀티모달 사전훈련 모델은 정렬된(multi-modal pairs 또는 tuples) 데이터에 지나치게 의존하는 경향이 있음. 데이터가 이미 모달리티 간에 정렬되어 있어야 하며, 일반적으로 시각적 정보 또는 음성 정보가 텍스트 정보와 함께 제공됨. 이는 큰 규모의 응용 프로그램에서 비용이 많이 들 수 있으므로, 정렬되지 않거나 짝이 없는 멀티모달 데이터를 사용하는 방법이 연구되고 있음(cross-modal supervision)
- Most of the existing pretext tasks transfer well across modalities.
    - pretext tasks: 모델이 특정 문제를 해결하기 위한 훈련을 진행하기 전에 수행하는 보조 작업
    - Masked Language Modeling(MLM)은 주로 텍스트 데이터에서 사용되지만, 오디오 및 이미지 데이터에 대해서도 적용 가능함
    - Frame Ordering Modeling(FOM)도 비디오 도메인에서 사용되지만, 텍스트 도메인에서도 유사한 아이디어를 활용해 작업할 수 있음
        - FOM: 비디오 프레임(영상에서 연속적인 이미지 프레임)의 순서를 예측하는 것이 핵심 아이디어
- 모델 구조: 멀티모달 사전훈련 모델은 주로 self-attention의 변형을 기반으로 하며, single-stream, multi-stream, hybrid-stream의 세 가지 범주로 나뉨
- cross-modal interactions: 다양한 모델 구조와 사전훈련 파이프라인에서 다양한 방식으로 cross-modal interactions를 학습할 수 있음. transformer 모델을 사용하여 이러한 상호작용을 학습함.
    - 멀티모달 사전학습은 다양한 구성 요소로 구성됨. 이러한 구성 요소에는 tokenization, transformer representation, objective suprevision이 포함될 수 있음. 각 요소는 멀티모달 데이터의 처리 및 학습에 기여함
    - self-attention은 임의의 모달리티로부터 임의의 토큰을 그래프의 노드로 임베딩하여 모델링하는 방식. 이것은 모달리티 간의 목적을 고려하지 않고 모달리티 간에 독립적으로 사용될 수 있음

**Task-Specific Multimodal Pretraining**

- 기존 기술의 제한으로 인해 모든 다양한 down-stream 응용에서 작동하는 매우 보편적인 네트워크 architecture, pretext task 및 corpora(말뭉치) 셋을 설계하는 것은 극히 어려움

### Transformers for Specific Multimodal Tasks

- Multimodal Inputs Encoding: transformer 모델은 최근 연구에 따르면 기존 및 새로운 discriminative 작업에서 다양한 multimodal inputs를 인코딩할 수 있다는 것을 보여줌. 예를 들면 RGB 이미지와 optical flow, RGB 이미지와 depth 정보, RGB 이미지와 point cloud 데이터, RGB 이미지와 LiDAR 데이터 등과 같이 다양한 모달리티 간의 조합을 다룰 수 있다는 것을 의미함.
- Multimodal Generative Tasks: transformer는 single-modality to single-modality를 포함한 다양한 multimodal generative task에도 기여함
    - RGB to scene graph, graph to graph, knowledge graph to text 등

## Challenges and Designs

### Fusion

- 일반적으로 Multimodal (MML) Transformer는 주로 early fusion(입력 수준의 융합), middle fusion(중간 수준의 융합), late fusion(출력 수준의 융합)의 세가지 수준에서 여러 양식에 걸쳐 정보를 융합함
- 일반적인 초기 fusion 기반 MML Transformer 모델은 “one-stream architecture”라고도 알려져 있으며, 최소한의 architecture 수정으로 BERT의 장점 채택 가능
- one-stream model의 주요 특징은 problem-specific modalities와 variant masking techniques를 사용한다는 것

### Alignment

- cross-modal alignment는 다양한 현실 세계의 멀티모달 응용 분야에서 중요한 역할을 하는 개념. 다양한 유형의 데이터 또는 모달리티 간의 조정과 일치를 의미하며, 주로 텍스트, 이미지, 음성, 비디오와 같은 다양한 모델 데이터 간의 관계를 말함.
    - transformer based cross-modal alignment: 다중 화자 비디오의 화자 위치 추정, speech translation, text-to-speech alignment(타임스탬프 작업 등에 활용) 등
- transformer를 기반으로 한 cross-modal alignment model은 주로 다음과 같은 방식으로 작동함
    - common representation space: 서로 다른 두 모달리티를 common representation space에 매핑
    - 쌍을 이루는 샘플: 일반적으로 모델은 서로 다른 두 모달리티를 가진 데이터 샘플 쌍에 대해 학습 (contrastive learning)
    - 이러한 cross-modal alignment model은 대부분 큰 모델이며, 대량의 훈련 데이터와 최적화에 많은 비용이 듬
    - 최근에는 cross-modal alignment model을 사전 훈련하여, 다양한 downstream task를 처리하고 있음. 이러한 모델은 zero-shot transfer를 가능하게 함

### Transferability

- transferability는 모델이 하나의 작업이나 데이터셋에서 학습한 지식을 다른 작업이나 데이터셋으로 전이하거나 활용할 수 있는 능력
- data augmentation, adversarial perturbation(적대적 변조) 전략은 multimodal transformer가 일반화 능력을 갖추는데 도움이 됨(VILLA, CLIP 등)
    - VILLA (Vision, Language, and Layout): 이미지와 텍스트 간의 상호 작용을 모델링하는데 중점을 둔 모델
    - CLIP (Contrastive Language-Image Pretraining): OpenAI에서 개발한 모델로, 텍스트와 이미지 간의 contrastive learning을 사용하여 텍스트와 이미지 간의 관계 학습
- transformer는 overfitting 가능성이 있어 최근 일부 사례에서는 noise가 없는 데이터셋에서 훈련된 oracle 모델(특정 작업을 완벽하게 수행하는 가상의 이상적인 모델 또는 시스템)을 실제 데이터셋으로 transfer 하는 방법을 활용
- cross-task gap
    - 멀티모달 모델이 다른 작업 간에 얼마나 잘 전이되는지에 대한 어려움
    - data augmentation, knowledge distillation 등을 사용하여 해결
- cross-lingual gap
    - 멀티모달 모델이 다른 언어 간에 얼마나 잘 전이되는지에 관한 것
    - ex. English → non-English

### Efficiency

- multimodal transformer model의 효율성 문제
    - 모델 크기와 데이터 의존성: 멀티모달 트랜스포머 모델은 대규모 훈련 데이터에 의존하고, 모델의 매개변수 크기가 크기 때문에 계산 및 메모리 요구 사항이 증가함
    - 시간 및 메모리 복잡성: 트랜스포머 모델의 self-attention은 입력 시퀀스의 길이에 따라 시간 및 메모리 복잡성이 제곱으로 증가하는 경향이 있어 대규모 데이터셋 및 긴 시퀀스 처리에 어려움이 있음
- 문제 해결을 위한 아이디어와 전략
    - knowledge distillation: 큰 트랜스포머 모델로 훈련된 정보를 더 작은 트랜스포머 모델로 전달하여 모델 크기를 줄이고 효율성을 높임
    - 모델 단순화 및 압축: 모델 구성 요소를 제거하거나 단순화해 모델의 복잡성을 낮추고 효율성을 높임
        - multimodal transformer에서 optimal sub-structures/sub-networks를 탐색
        - VLP transformer
            - 2-stage pipeline은 object detector가 필요해서 비용이 많이 듬 → VLP, ViLT와 같이 convolution 없는 방식으로 시각적 입력 처리
        - DropToken
            - 훈련 중 무작위로 토큰의 일부 삭제
        - weight 공유
    - asymmetrical network structures: 다양한 양식에 대해 다양한 모델 용량과 계산 크기를 적절하게 할당하여 매개변수 저장
    - training sample의 활용
        - 데이터를 다양한 granularity(세분성) 수준에서 활용. 세분성이란 데이터의 다양한 측면이나 세부 정보 수준으로, 이미지 데이터의 경우 세분성 수준을 낮추면 이미지의 전체 내용을 고려하는 반면, 높게 설정하면 이미지 내의 작은 세부 사항을 고려함
        - 더 적은 양의 훈련 데이터를 활용하면서도 모델을 효과적으로 훈련시키는 것이 목표
        - ex) CLIP 훈련
            - self-supervised learning과 multi-view supervision(여러 관점에서의 학습)을 활용하여 데이터의 다양한 세분성과 정보를 활용
            - 다른 유사한 쌍에서 nearest-neighbor supervision 활용
    - self-attention 복잡성 최적화: self-attention의 복잡성을 줄이기 위해 sparse factorization(희소 인수분해) 등의 방법을 사용하여 효율성 향상
    - self-attention 기반 multimodal interaction/fusion의 복잡성 최적화
        - 일반적으로 다중 모달 데이터를 처리하는 모델에서는 다양한 모달리티 간의 interaction과 정보 융합이 필요함. 이러한 상호작용 및 융합은 모델의 복잡성을 높일 수 있고, 계산 비용을 증가시킬 수 있음
        - early concatenation은 모델의 크기가 커질수록 계산 비용이 증가하고 메모리 사용량이 높아질 수 있기 때문에 Fusion via Attention Bottleneck을 제안
            - Fusion via Attention Bottleneck: self-attention 매커니즘을 사용하여 다중 모달 데이터 간의 상호작용 및 융합을 수행하는데, 이때 Fusion Bottleneck이라는 특별한 구성 요소를 활용함. Fusion Bottleneck은 상호작용 및 융합 단계에서 중요한 정보만을 선택적으로 고려하고, 불필요한 정보를 필터링해 계산 비용을 최적화하고 메모리 사용량을 줄이는 역할을 함
    - 기타 전략: Greedy strategy을 사용하여 시퀀스 길이의 오름차순으로 정보를 융합하는 방법 등

### Robustness

- robustness: 모델이 다양한 상황에서 안정적으로 작동하고 오류나 이상 동작을 최소화하는데 관련된 개념
- 대규모 데이터셋에서 pretrained된 multimodal transformer 모델들은 SOTA(최고 수준의) 성능을 보이지만, 여전히 robustness 측면에서 불명확한 부분이 있음
- 이론적으로 robustness를 분석하고 개선하는 방법에 대한 연구 필요
- 최근 연구
    - transformer components/sublayer가 robustness에 어떻게 기여하는지 연구 및 평가
    - 하지만, transformer 모델을 분석하기 위한 이론적 도구가 부족함
    - robustness를 평가하는 일반적인 방법은 다른 데이터셋 간의 평가 또는 데이터에 미세한 변화를 적용하여 모델의 반응을 살펴보는 perturbation 기반 평가 등이 있음
    - data augmentation과 adversarial training 기반 전략 시도
    - fine-grained loss function(더 세부적인 손실 함수) 사용 시도

### Universalness

- 최근 많은 연구가 다양한 양식과 멀티모달 작업을 처리하기 위한 통합된 파이프라인을 사용하는 방법을 연구
- 최근 시도
    - uni-modal과 multimodal 입력 및 작업을 위한 파이프라인 통합
    - multimodal 이해(understanding)와 생성(generation)을 위한 파이프라인 통합
        - 일반적으로, multimodal transformer 파이프라인은 understanding 및 discriminative task를 처리하는데 필요한 부분에서만 encoder를 사용. 반면 generation 또는 generative task의 경우 encoder와 decoder 모두 필요
        - 기존 시도는 multitasking learning을 사용하여 understanding과 generation workflow를 결합해 multitask loss funciton에 의해 공동으로 학습되도록 함
    - CLIP
        - 이미지와 텍스트를 모두 이해하고 다룰 수 있는 모델로, 작업 자체를 통합하고 변환함으로써 “zero-shot recognition”을 “retrieval”로 변환하는데 사용됨
        - 모델이 주어진 이미지나 텍스트에 대한 이해력을 사용하여 관련된 쿼리를 검색하고 관련성을 평가하는데 활용됨
- 위 시도들의 과제 및 bottlenecks
    - universal model의 보편성과 비용 균형: 학습 및 추론 비용이 증가할 수 있으며, 특정 작업에 대한 최적화를 희생할 수 있음. 이로 인해 모델의 성능을 향상시키는데 일부 제약이 따를 수 있음
    - multi-task loss function의 복잡성 증가: 다중 작업 학습을 위해 사용되는 multi-task loss function은 모델을 더 복잡하게 만들 수 있음

### Interpretability

- transformer가 multimodal learning에서 왜 잘 수행되는지와 그 방법 해석
- interpretability를 높이기 위한 다양한 기술과 도구: attention map 시각화, 각 모달리티의 기여도 분석, 무델의 중요한 입력 특징 식별 및 설명, 설명 가능한 모델의 개발 등

## Discussion and Outlook

- MML 모델을 다양한 모달리티에서 탁월한 성과를 내도록 디자인하는 것은 어려운 도전임
- 2 stream architecture는 cross-modal retrieval 작업에 효율적이지만, modal-agnostic MML 아키텍처를 설계하는 것은 여전히 어려움
- 최신 기술과 MML 모델 간에는 여전히 큰 차이가 있음. 이를 줄이고 모든 작업에 적용 가능한 모델 디자인을 탐구하기 위한 연구가 계속 진행 중
- 모델 디자인과 계산 비용 등 여러 요인을 고려해야 하며, 산업 연구팀들이 이 분야에서 활발한 연구를 이끌 것으로 예상
- transformer의 장점
    - implicit knowledge(모델이 데이터로부터 학습한, 명시적으로 주어지지 않은 지식)를 인코딩할 수 있음
    - multi-head 구조로 representation 능력을 향상시킬 수 있음
    - 본질적으로 non-local 패턴을 인식하는 global aggregation의 성격을 가지고 있음
    - 대규모 데이터에 대한 효과적인 사전 학습을 통해 도메인 간 차이를 다루기 용이함
    - 다양한 입력 양식과 호환 가능
    - 시리즈 및 시퀀스 패턴(시계열) 모델링의 경우 훈련 및 추론의 병렬 계산 덕분에 RNN 기반 모델에 비해 더 나은 훈련 및 추론 효율성을 가짐
- 토큰화를 사용하면 transformer가 multimodal input을 유연하게 구성할 수 있음

## Conclusion

- 이 survey는 transformer를 사용한 multimodal machine learning에 중점을 둠