from pathlib import Path
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1)
base = Path(__file__).resolve().parent
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt

out = base / "outputs"
out.mkdir(exist_ok=True)
# 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.
device = torch.device("cuda:0")
df = pd.read_csv(base / "data" / "points.csv")
# CSV 안에 이미 split 열이 있어서, 그 값으로 train 240건과 valid 80건을 골라냅니다. test 80건은 아예 읽지 않습니다.
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
# Tensor 변환. x1, x2는 float32, label은 long(정수)입니다. CrossEntropyLoss가 정답을 정수 클래스 번호로 받기 때문입니다. 
# valid만 .to(device)를 미리 하고 train은 안 하는 이유는, train은 DataLoader로 배치마다 꺼내면서 55행에서 옮기기 때문입니다.
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)
# 손실 함수. 2클래스 분류라 CrossEntropy입니다.
loss_fn = nn.CrossEntropyLoss()
# 후보가 lr 0.01은 공통이고, sgd는 optimizer만, adamw_dropout은 dropout 0.3만 추가로 다릅니다.
# weight decay를 gradient에 섞느냐(Adam) 따로 빼서 적용하느냐(AdamW)
configs = [{'name': 'sgd', 'optimizer': 'SGD', 'lr': 0.01}, {'name': 'adamw', 'lr': 0.01}, {'name': 'adamw_dropout', 'lr': 0.01, 'dropout': 0.3}]
summary = []

for config in configs:
    torch.manual_seed(42)
    width = config.get("width", 32)
    layers = [nn.Linear(2, width)]
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])
    model = nn.Sequential(*layers).to(device)
    # shuffle용 Generator를 후보마다 같은 seed로 다시 만듭니다.
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)
    # 240건을 32개씩 섞어서 꺼냄 → 한 epoch에 8번(마지막은 16건)
    # generator seed 43: 데이터 섞는 순서도 세 후보가 똑같게
    if config.get("optimizer", "AdamW") == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                    momentum=config.get("momentum", 0.0), weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, 40 + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            # loss.backward() 다음 optimizer.step()을 호출하세요.
            # 같은 초기화·동일 shuffle 순서를 사용하며 세 후보의 변경 조건을 읽어 구분합니다.
            # >>> A07
            # TODO A07: 위 요구사항에 맞게 이 블록만 완성합니다.
            loss.backward()
            optimizer.step()
        # 같은 epoch 말 모델을 eval 모드로 비교합니다. 학습 중 Dropout loss와 구분합니다.
        # epoch 종료 후 model.eval()을 호출하세요.
        # 그 아래 no_grad 블록은 gradient 추적을 끄지만, Dropout 모드를 바꾸지는 않습니다.
        # TODO A08: 위 요구사항에 맞게 이 블록만 완성합니다.
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            logits = model(X_valid) # valid 80건 → (80, 2). 클래스별 점수(softmax 전 값)
            valid_loss = loss_fn(logits, y_valid).item()
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
            # 최저 valid loss를 관측하면 값·epoch·가중치 복사본을 함께 보관합니다.
            # A09 요구: 동률은 먼저 발견한 epoch 유지. copy.deepcopy(model.state_dict()) 사용.
            # model.state_dict()만 대입하면 이후 학습이 같은 Tensor를 바꾸므로 안 됩니다.
            # TODO A09: 최적 loss·epoch와 독립된 state_dict 복사본을 보관하세요.
            if valid_loss < best_loss:                          # 지금까지 중 최저인가
                best_loss = valid_loss                          # 점수 기억
                best_epoch = epoch                              # 몇 번째였는지 기억
                best_state = copy.deepcopy(model.state_dict())  # 가중치 사진
            # <<< A09
        history.append([epoch, train_loss, valid_loss, valid_acc])
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]
    table.to_csv(out / f"07_{name}_history.csv", index=False)
    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})
    # 제공 코드: 학습이 끝난 뒤 최적 상태로 복원하고 그 모델로 다시 평가합니다.
    model.load_state_dict(best_state) 
    model.eval() 
    with torch.no_grad():
        best_pred = model(X_valid).argmax(1).cpu()
        restored_loss = loss_fn(model(X_valid), y_valid).item()
    labels = y_valid.cpu()
    class_rows = []
    for label in (0, 1):
        # A10 요구: 실제 정답이 label인 행을 먼저 고른 뒤 support·correct·recall 계산.
        # 전체 accuracy나 '예측 label의 수'로 나누지 않습니다. 0건이면 recall=None.
        # >>> A10
        # TODO A10: 클래스별 실제 표본 수·정답 수·Recall을 기록하세요.
        mask = labels == label                            # 실제 정답이 label인 자리만 True (80개 중)
        support = int(mask.sum())                         # True 개수 = 실제로 이 클래스인 건 수 → Recall 분모
        correct = int((best_pred[mask] == label).sum())   # 그 자리의 예측 중 label과 같은 수 = 맞힌 수(TP)
        recall = correct / support if support > 0 else None   # 실제 수로 나눈 비율. 0건이면 None
        class_rows.append({"label": label, "support": support,     # 클래스 하나 = 표 한 행
                           "correct": correct, "recall": recall})  # 두 바퀴 돌면 2행 → CSV로 저장
        # <<< A10

# ┌─────────┬──────────────────────────┬───────────────────────────────────────────────────┐
# │  이름   │            뜻            │                       코드                        │
# ├─────────┼──────────────────────────┼───────────────────────────────────────────────────┤
# │ support │ 실제로 이 클래스인 건 수 │ mask = labels == label; support = int(mask.sum()) │
# ├─────────┼──────────────────────────┼───────────────────────────────────────────────────┤
# │ correct │ 그중 모델이 맞힌 수      │ correct = int((best_pred[mask] == label).sum())   │
# ├─────────┼──────────────────────────┼───────────────────────────────────────────────────┤
# │ recall  │ correct ÷ support        │ correct / support if support > 0 else None        │
# └─────────┴──────────────────────────┴───────────────────────────────────────────────────┘

    pd.DataFrame(class_rows).to_csv(out / f"07_{name}_class_metrics.csv", index=False)
    pd.DataFrame({"id":valid["id"].tolist(),"actual":labels.tolist(),
                  "predicted":best_pred.tolist()}).to_csv(out / f"07_{name}_predictions.csv", index=False)
    summary[-1]["restored_valid_loss"] = restored_loss
    summary[-1]["restored_matches_best"] = abs(restored_loss-best_loss) < 1e-6
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"07_{name}_curves.png", dpi=150)
    plt.close()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))
result = pd.DataFrame(summary)
result.to_csv(out / "07_comparison.csv", index=False)
print("생성: outputs/07_*.csv, *.png / test 미사용")
