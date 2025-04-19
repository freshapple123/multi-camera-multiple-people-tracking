# Multi-camera Multiple People Tracking (real-time)
계속해서 실험중이고 실험중간에 저장소로 사용중입니다.  
I am still experimenting and using it as a repository during the process.  

---
## 🔍 Re-ID 실험 중간 결과 (real-time) (25.04.20)

### 📌 Single-Angle Re-ID (real-time)
- **방식**: 단일 각도에서 촬영된 영상 기반  




https://github.com/user-attachments/assets/90fb69bd-4747-4182-8ae0-57bc993dcca4




---

### 🎥 Multi-Angle Re-ID (real-time) 

#### ✅ ID = 0 (다각도 추적 결과)


https://github.com/user-attachments/assets/f98deac0-4ee7-4fe8-927e-234a00285b13



#### ✅ ID = 1 (다각도 추적 결과)



https://github.com/user-attachments/assets/b37e6b83-4d14-4901-884c-7a6118b19711


## 📄 초록 (Abstract)

### 🇰🇷 한국어

최근 사람 재식별(Person Re-Identification, Re-ID) 연구는 주로 딥러닝 기반 모델을 활용하여 고차원 특징을 추출하고 이를 비교하는 방식으로 이루어지고 있다. 이러한 방법은 높은 정확도를 보이지만, 연산량이 많아 실시간 적용이 어렵고, 임베디드 시스템과 같이 제한된 연산 자원을 가진 환경에서는 활용에 한계가 있다. 본 연구는 히스토그램 기반의 영상처리 기법을 활용하여 실시간으로 사람을 재식별하는 방법의 가능성을 탐색하였다.

히스토그램 기반 방식은 계산량이 적고 경량화가 용이하여 실시간 시스템에 적합하며, 딥러닝을 사용하지 않고도 일정 수준의 구분력을 확보할 수 있다는 장점을 가진다. 하지만 기존 연구에서는 정확도가 낮고 조명 변화나 다양한 포즈, 시점 변화에 취약하다는 문제가 있었다.

본 연구에서는 이러한 단점을 보완하기 위해, 경량화된 AI 모델을 활용해 주요 영역을 사전에 특정한 후, 영상처리 기반의 히스토그램 비교 방식으로 최종 판별을 수행함으로써 연산 부담을 줄이면서도 실시간 적용이 가능한 재식별 시스템을 제안한다.

이러한 접근은 CCTV 기반의 실시간 분석이 요구되는 상황에서 학습 기반보다 규칙 기반(Rule-based) 방식이 더 현실적인 대안이 될 수 있으며, 제한된 환경에서도 효과적으로 사람을 재식별할 수 있는 가능성을 제시한다.

---

### 🇺🇸 English

Recent research on Person Re-Identification (Re-ID) primarily relies on deep learning-based models to extract and compare high-dimensional features. While these methods achieve high accuracy, they require significant computational resources, making real-time applications difficult, especially in environments with limited processing capabilities such as embedded systems. This study explores the feasibility of using histogram-based image processing techniques for real-time person re-identification.

Histogram-based methods offer low computational cost and are lightweight, making them suitable for real-time systems. They can provide a reasonable level of distinction without relying on deep learning. However, conventional approaches using this method often suffer from low accuracy and are vulnerable to changes in lighting, poses, and viewpoints.

To address these limitations, this study proposes a real-time re-identification system that utilizes a lightweight AI model to pre-locate key regions, followed by a histogram-based comparison for final identification. This approach reduces the computational load while maintaining real-time applicability.

Such a method suggests that in scenarios requiring real-time CCTV analysis, rule-based methods can be a more practical alternative to learning-based approaches, offering the potential for effective person re-identification even in resource-constrained environments.


## 📚 Dataset Citation

If you use this dataset in your research, please cite the following paper:

**MMPTRACK: Large-scale Densely Annotated Multi-camera Multiple People Tracking Benchmark**  
*Xiaotian Han, Quanzeng You, Chunyu Wang, Zhizheng Zhang, Peng Chu, Houdong Hu, Jiang Wang, Zicheng Liu*  
arXiv:2111.15157

BibTeX:
```bibtex
@misc{han2021mmptrack,
    title={MMPTRACK: Large-scale Densely Annotated Multi-camera Multiple People Tracking Benchmark}, 
    author={Xiaotian Han and Quanzeng You and Chunyu Wang and Zhizheng Zhang and Peng Chu and Houdong Hu and Jiang Wang and Zicheng Liu},
    year={2021},
    eprint={2111.15157},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
