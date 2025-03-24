# 사람 ID별 저장소 예시
person_db = {
    id_0: {
        "histories": [hist1, hist2, ...],     # HS 히스토그램 누적
        "last_position": (x, y),              # 최근 위치 (ex. 중심점)
        "frames_since_seen": 0                # 마지막 감지된 이후 프레임 수
    },
    id_1: {
        ...
    },
    ...
}

next_id = 0  # 새로운 사람을 위한 ID 카운터



# 히스토리 관리 전략 (예시)
def get_hist_average(histories, N=5):
    if len(histories) == 0:
        return None
    recent = histories[-N:]
    return np.mean(recent, axis=0)


