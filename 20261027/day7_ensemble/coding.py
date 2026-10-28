
N = "지민이의 소"
M = "지민이의 말"

C1 = "지민이의 소 N마리의 x 좌표"
C2 = "지민이의 말 M마리의 x 좌표"

# P=(x1, y1, z1)와 Q=(x2, y2, z2) 간의 거리는 dist (P, Q) = |x2-x1 |+|y2-y1 |+|z2-z1 | 로 정의된다.



# 모든 소와 말 쌍에 대해서, 가장 가까운 쌍의 거리와, 이러한 최소 거리를 가지는 쌍의 개수


# 첫 테스트 케이스
"""
소 3마리, 말 4마리
소들은 (1, 0), 말들은 (-3, 0)

소 (1, 0, 3), (1, 0, 0), (1, 0, 6)
말 (-3, 0, -2), (-3, 0, 5), (-3, 0, 4), (-3, 0, 2)
"""

# 모든 조합 만들기

def dist(p, q):
    return abs(q[0] - p[0]) + abs(q[1] - p[1]) + abs(q[2] - q[2])

T = int(input())
for test_case in range(1, T + 1):
    n, m = map(int, input().split())
    c1, c2 = map(int, input().split())

    cows = []
    for z in input().split():
        cows.append((c1, 0, int(z)))

    horses = []
    for z in input().split():
        horses.append((c2, 0, int(z)))

    min_dist = float("inf")
    count = 0

    for cow in cows:
        for horse in horses:
            d = dist(cow, horse)
            if d < min_dist:
                count = 1
            elif d == min_dist:
                count += 1

    print(f"#{test_case} {min_dist} {count}")