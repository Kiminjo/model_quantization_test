from typing import List, Dict

# 학습할 클래스 목록
CLASSES: List[str] = [
    'FM'
]

# 클래스 이름을 인덱스로 매핑
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

# 생성할 패치 사이즈 목록 (다양한 크기 지정 가능)
PATCH_SIZES: List[int] = [400, 512, 1024]

# 패치 생성 시 겹치는 비율
OVERLAP_RATIO: float = 0.2
