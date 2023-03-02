## ‼️ 기획 의도
자동 통화 녹음, 꺼야할까? 켜야할까?
- 😊 통화 녹음의 장점: 보이스피싱, 협박, 괴롭힘 등 예상치 못한 상황에서 객관적 자료 확보
- 😒 통화 녹음의 단점: 일상대화, 스팸 등 불필요한 통화 내용까지 모두 저장

→ 쌓여가는 통화녹음, **필요한 통화녹음만** 저장할 수 있는 서비스 개발

---

## 📝 프로젝트 Summary
### AI 기반 음성 데이터를 통한 '통화 녹음 저장 판단' 시스템

---

## 📞 앱 이미지
![merge_from_ofoct](https://user-images.githubusercontent.com/74871378/221477298-cc9d97bc-b31a-4a6b-93f9-93aea51fe226.png)

---

## 🚩 개발 프레임워크
![image](https://user-images.githubusercontent.com/71757818/221478426-f6a518d6-5449-4ea2-82df-d56e737db08e.png)

---

## 🌐 개발 환경
#### 💻 Python
- 머신러닝 AI 모델 개발을 위한 프로그래밍 언어
 
#### 📱 Android Studio
- SKT 연계 프로젝트로서, T전화 앱의 서비스 추가를 목표로 Android 개발 환경 채택
- 프로그래밍 언어 Java

#### 🖼️ Figma
- Android UI와 Color Chip 구상

#### ☁️ Azure Cloud
- 클라이언트와 통신하기 위한 서버
- 병렬화 가상머신 통한 모델 동작

---

## ⚡ 감정 분류 모델
▶ 음성 감정 분류

✔ wav2vec2.0
- Facebook wav2vec2.0 전이학습 모델
- Self-Supervised Learning
- Fine-Tuning
- Positive와 Angry 잘 분류
- Accuracy: 0.75

▶ 텍스트 감정 분류

✔ BERT
- 앞 뒤 문맥을 모두 고려할 수 있는 양방향 Encoder 구조
- WordPeice 토크나이저를 subword 식별 → 신조어, 일상어 식별↑
- Positive와 Fear 잘 분류
- Accuracy: 0.76

▶ 음성 + 텍스트 멀티 모달 

✔ wav2vec2.0 + BERT
- 각 모델이 잘 감지하는 감정에 따라 가중치 부여
- Accuracy: 0.82

---

## ⚡요약 모델
▶ **첫 번째 시도**
- 길이가 긴 요약 데이터 제거
- 문서를 BART를 통해 요약

✔ 첫 번째 시도: 결과
- 애매한 성능
- STT 성능상의 단어 오류 존재
- 오타, 불용어 등 부적절한 단어들이 요약에 그대로 포함
- 문장 종료 토큰을 지정했음에도 문장이 마침표로 끝나지 않고 중간에 끊긴 문장이 등장

▶ **두 번째 시도**
- Embedding 벡터의 크기가 너무 작아 원문, 요약문의 특성을 모두 담지 못함 → 원문, 요약문의 텍스트 크기 분포를 각각 확인하여 Embedding 벡터 수정
- Kiwi 형태소 분석기를 사용 → 단어 오류, 맞춤법 교정, 불용어 제거 
- 마침표로 끝나지 않는 문장 제거

✔ 두 번째 시도: 결과
- 불필요한 단어들이 제거된 탓에 요약 문장이 매끄럽지 않음 → Kiwi 형태소 분석기 사용 기각
- 마침표로 끝나는 온전한 문장들만 등장
- 길이가 너무 짧아 의미가 없는 문장이 등장 + 문장 수가 많은 요약문 등장

▶ **세 번째 시도**
- 너무 짧은 문장 제거, 문장 개수 제한

✔ 세 번째 시도: 결과
- 한 두 문장으로 구성된 요약문 생성 성공

---

## ⚡키워드 모델
▶ **첫 번째 시도**
- 문서에서 Sklearn의 CountVectorizer를 사용하여 키워드 추출
- 문서와 문서로부터 추출한 키워드를 BERT를 통해 Embedding
- Cosine Similarity를 사용해 문서와 가장 유사한 키워드 추출

✔ 첫 번째 시도: 결과
- STT 성능상의 단어 오류 존재
- 동사형 등 키워드에 부적합한 단어 등장
- 비슷한 의미의 키워드들이 리턴

▶ **두 번째 시도**
- Kiwi 형태소 분석기를 사용 → 단어 오류, 맞춤법 교정, 불용어 제거, 명사만 추출

✔ 두 번째 시도: 결과
- 자연스러운 키워드 추출 성공
- 여전히 비슷한 의미의 키워드들이 리턴

▶ **세 번째 시도**
- MMR(Maximal Marginal Relevance) 사용 → 중복을 최소화하고 결과의 다양성 극대화<br>
       1) 문서와 가장 유사한 키워드 선택<br>
       2) 문서와 유사하고 이미 선택된 키워드와 유사하지 않은 새로운 키워드를 반복적으로 선택 

✔ 세 번째 시도: 결과
- 주제 관련성이 높은 키워드 추출
- 키워드 중복을 방지하여 다양한 키워드 추출

---

## 🧑‍💻역할
**성시열**(https://github.com/sungsiyul)
- Product Manager
- 음성 감정 분류 모델
- Cloud 서버 VM 네트워킹

**김지환**([https://github.com/hongju904](https://github.com/hongju904))
- Android 앱 UI/UX 개발 ([구현 내용](https://velog.io/@hongju904/VoiceKeeper-app-%EA%B0%9C%EB%B0%9C-%EA%B8%B0%EB%A1%9D))
- Cloud 서버와 Android 클라이언트 HTTP 통신

**전민기**(https://github.com/mk9165)
- 텍스트 감정 분류 모델
- 대화 요약, 키워드 추출

**최형준**(https://github.com/junhjun)
- 음성 감정 분류 모델
- 음성 + 텍스트 멀티 모달
- Cloud 서버 VM 네트워킹

---

## 🧐 문제점 및 해결 내용
### 🔊 음성 감정 분류

1️⃣ **첫 번째 시도: 특징 벡터 모델**
- 음성에서 같은 시간 간격의 특징 벡터 추출
- Zero Crossing Rate, Chroma STFT, Mel Spectrogram, MFCC, RMS
- 1D CNN
- [Reference](https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition)

🔍 **결과**
- 특정 감정의 민감도 낮음(angry, sad)
- 특징 벡터 추출 시 오랜 시간 소요 → 특징 벡터 추가 실패

2️⃣ **두 번째 시도: 스펙트로그램 이미지 모델**
- 음성에서 같은 시간 간격의 스펙트로그램 이미지 추출
- 시간과 주파수에 따른 진폭의 변화 시각화 기법
- 2D CNN(VGG13)
- [Reference](https://github.com/HanNayeoniee/Trauma-Detector)

🔍 **결과**
- 전반적인 정확도 낮음
- 이미지 추출 시 오랜 시간 소요
- 층이 깊은 모델 구조 → 과적합 발생

3️⃣ **세 번째 시도: wav2vec2.0**
- 자기지도학습으로 대량 데이터 학습한 효과
- [Reference](https://huggingface.co/harshit345/xlsr-wav2vec-speech-emotion-recognition)

🔍 **결과**
- 감정 분류의 정확도 및 분석속도 개선
- 라벨링 된 음성 데이터의 부족함 해결
