# 2. Debugging challenge (PyTorch==1.1.0)


**Run:** `python train.py --(args)`
처음에 한번 돌리시면 `data/glove`, `data/news20` 디렉토리가 생성되고 종료됩니다.


## Dataset

**20 Newsgroup Dataset** 에 속한 문서들에 대해 multi-class classification을 진행합니다. (데이터셋에 대한 설명: [scikit-learn link](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html#newsgroups) 참고) 

원래 데이터셋은 클래스가 20개지만, 우리는 4개의 클래스만 추출하여 진행합니다. 사용하는 클래스는 `sci` 계열에 속한 주제들입니다.

* sci.crypt
* sci.electronics
* sci.med
* sci.space

추출된 데이터셋은 매우 작습니다 (<3K). 최초 훈련시 데이터가 `data/news20` 자동으로 폴더에 저장될 것입니다.



## Vocabulary

Vocabulary로는 [pretrained glove](https://nlp.stanford.edu/projects/glove/) 중 Wikipedia 2014 + Gigaword 5에 대해서 학습된 glove.6B.100d.txt에 있는 단어들을 사용하겠습니다. 원래 vocab은 400K개지만, 우리는 top 50K만 추출하여 진행합니다. + PAD, UNK  토큰을 두어서, vocab size=50,002가 됩니다. 

**주의 사항.** 현재 코드에는 glove에 있는 단어의 *목록* 만 사용하고, pretrained weights는 사용하지 않습니다.

**할 일.** glove.6B.100d.txt 를 구하셔서  `data/glove` 폴더 아래에 집어넣어 주시면 됩니다.



## 모델

Yang et al. (2016). Hierarchical Attention Networks for Document Classification ([링크](https://www.cs.cmu.edu/%7Ediyiy/docs/naacl16.pdf)) 논문의 HAN 모델을 구현하였습니다. 버그가 가득한 코드입니다. 트레이닝 로스가 기이하게도 음수로 찍히고 있습니다!



## Tasks (총 10pt + 2pt bonus)


### 1. (8pt) 버그/오류 잡기

어느 부분들을 고쳤는지, 왜 그렇게 고쳤는지 report에 명시해 주시고, 코드에도 `NOTE:` 와 같이 **화려한... 찾기 쉬운** 주석을 달아 주세요. :)

* `dataset.py` 에는 오류가 없다고 가정하셔도 됩니다.

* 나머지 4개의 파일 (`{dataloader, model, train, trainer}.py`) 에는 오류가 있을 수 있습니다.

* 고치지 않아도 성능에는 문제없어 보이는 오류도 있습니다.

  

### 2. (1pt) Evaluation (test time) 코드 작성하기

* 테스트용 dataset을 사용해야 합니다. 

* 코드를 run하는 방법을 report에 명시해주세요. 



——— 여기까지 제대로 수정하셨다면, 3 epoch이 지난 후 test time accuracy가 0.5 이상 나옵니다.


### 3. (1pt) Pretrained embedding 적용하고 fine-tune될 수 있도록 하기

* glove.6B.100d 버전을 적용해 주시면 됩니다.

1) `model.py` 의 `class WordAttention` 에 비워 놓은 `init_embeddings` 및 `freeze_embeddings` 함수를 채워 넣고,

2) 이 두 함수들을 적절하게 모델에 적용하여 주시면 됩니다.

* Weights는 외부 라이브러리를 통해서 불러와도 되고 glove.6B.100d.txt 파일에서 직접 읽어들여도 됩니다. 외부 라이브러리를 사용하신 경우 report에 명시해 주세요.




### 4. (optional, +2pt) 모델 가지고 놀기

모델의 컨셉을 유지하는 선에서 원하는 대로 튜닝을 해 주시면 됩니다. 성능 향상에 초점을 둔 변화라면 성능을 증빙할 수 있는 자료와 함께 제출해주세요. 

* hidden_size, layer 수 등 구조의 세부적인 변화
* hyperparameter 튜닝
* pretrained representation, vocabulary 변경
* Efficient data loading
* Logger, Tensorboard 사용
* Attention weight visualization 등



### 제출 시 주의사항

1) 1+2번을 푼 코드를 베이스로 해서, 이후의 새끼문제에 대한 풀이들은 order를 유지하면서 increment하게 제출해주세요.

* 1,2번과 3번을 풀었다 = 1) 1+2번, 2) 1+2+3번을 적용한 버전 두 가지를 제출.
* 1,2번과 4번을 풀었다 = 1) 1+2번, 2) 1+2+4번을 적용한 버전 두 가지를 제출.
* 모두 풀었다 = 1) 1+2번, 2) 1+2+3번, 3) 1+2+3+4번을 적용한 버전 세 가지를 제출.

2) 3에폭까지의 성능을 보여줄 수 있는 자료도 함께 제출해주세요 (loss/accuracy 스크린샷, tensorboard graph 등).

3) 제출시에는 `data/glove`, `data/news20` 폴더를 없애고 제출해 주세요.