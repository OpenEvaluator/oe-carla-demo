import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

def generate_data(n=10000):
    X = []
    Y = []

    for _ in range(n):
        x_pos = np.random.uniform(0, 100)
        y_pos = np.random.uniform(0, 100)
        yaw = np.random.uniform(-180, 180)
        v_x = np.random.uniform(-10, 10)
        v_y = np.random.uniform(-10, 10)

        # Dummy control logic 
        steer = np.clip(yaw / 180, -1, 1)
        throttle = np.clip(1 - abs(v_x) / 10, 0, 1)
        brake = 1.0 - throttle if throttle < 0.3 else 0.0

        X.append([x_pos, y_pos, yaw, v_x, v_y])
        Y.append([steer, throttle, brake])

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

X, Y = generate_data(10000)


class MLModel(nn.Module):
    def __init__(self):
        super(MLModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Tanh()  # Steer ∈ [-1, 1]; throttle & brake ∈ [-1, 1], we'll clamp later
        )

    def forward(self, x):
        out = self.net(x)
        steer = out[:, 0:1]                     # shape (N,1)
        throttle = torch.clamp(out[:, 1:2], 0.0, 1.0)
        brake = torch.clamp(out[:, 2:3], 0.0, 1.0)
        return torch.cat([steer, throttle, brake], dim=1)


model = MLModel()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(30):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1:2d}, Loss: {loss.item():.6f}")

torch.save(model.state_dict(), "model.pt")
print("✅ Model saved as model.pt")

model = MLModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()
print("✅ Model loaded from model.pt")

def predict(model, state):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) 
        action = model(state_tensor)
        return action.squeeze(0).numpy()  

state = [50, 50, 0, 5, 0] 
action = predict(model, state)
print(f"Predicted action: steer={action[0]:.2f}, throttle={action[1]:.2f}, brake={action[2]:.2f}")

