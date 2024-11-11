from multiprocessing import Process

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from src import environment


class Hw1Env(environment.BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        r = np.random.rand()
        if r < 0.5:
            size = np.random.uniform([0.02, 0.02, 0.02], [0.03, 0.03, 0.03])
            environment.create_object(scene, "box", pos=[0.6, 0., 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=[0.8, 0.2, 0.2, 1], friction=[0.02, 0.005, 0.0001],
                                      density=4000, name="obj1")
        else:
            size = np.random.uniform([0.02, 0.02, 0.02], [0.03, 0.03, 0.03])
            environment.create_object(scene, "sphere", pos=[0.6, 0., 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=[0.8, 0.2, 0.2, 1], friction=[0.2, 0.005, 0.0001],
                                      density=4000, name="obj1")
        return scene

    def state(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return obj_pos, pixels

    def step(self, action_id):
        if action_id == 0:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 1:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 2:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 3:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})


def collect(idx, N):
    env = Hw1Env(render_mode="offscreen")
    positions = torch.zeros(N, 2, dtype=torch.float)
    actions = torch.zeros(N, dtype=torch.uint8)
    imgs = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    for i in range(N):
        action_id = np.random.randint(4)
        _, prev_image = env.state()
        env.step(action_id)
        obj_pos, _ = env.state()
        positions[i] = torch.tensor(obj_pos)
        actions[i] = action_id
        imgs[i] = prev_image
        env.reset()
    torch.save(positions, f"HW1/data/positions/positions_{idx}.pt")
    torch.save(actions, f"HW1/data/actions/actions_{idx}.pt")
    torch.save(imgs, f"HW1/data/images/imgs_{idx}.pt")


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.positions = []
        self.actions = []
        self.images = []

        # load from multiple files
        for i in range(4):
            self.positions.append(torch.load(f"HW1/data/positions/positions_{i}.pt"))
            self.actions.append(torch.load(f"HW1/data/actions/actions_{i}.pt"))
            self.images.append(torch.load(f"HW1/data/images/imgs_{i}.pt"))

        # concatenation
        self.positions = torch.cat(self.positions, dim=0)
        self.actions = torch.cat(self.actions, dim=0)
        self.images = torch.cat(self.images, dim=0)

        # sanity check
        assert len(self.positions) == len(self.actions) == len(self.images), "Data length mismatch"

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        ])

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        position = self.positions[idx]
        action = self.actions[idx]
        image = self.images[idx]
        image = self.transform(image)
        return position, action, image


# takes in the image of scene and executed action as inputs, and predicts the final object position as (x, y)
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(3 * 128 * 128 + 1, 256),  # 3 channels for RGB, 128x128 image, + 1 for action
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2)         # 2 output for x, y
        )

    def forward(self, x):
        return self.nn(x)

if __name__ == "__main__":

    # Data Collecting
    # processes = []
    # for i in range(4):
    #     p = Process(target=collect, args=(i, 250))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    # Model Training

    full_dataset = Dataset()
    train_size = int(0.8 * len(full_dataset))
    validation_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
                                                                                    [train_size, validation_size, test_size])

    dataLoader_training = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    dataLoader_validation = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)
    dataLoader_testing = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = DNN()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train(True)

    training_loss_graph = []
    validation_loss_graph = []

    for epoch in range(20):

        # training
        training_loss = 0
        for i, (positions, actions, images) in enumerate(dataLoader_training):
            optimizer.zero_grad()

            images = images.view(-1, 3 * 128 * 128)
            actions = actions.view(-1, 1)
            inputs = torch.cat((images, actions), dim=1)

            outputs = model(inputs)
            loss = criterion(outputs, positions)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        training_loss_graph.append(training_loss / len(dataLoader_training))

        # validation
        validation_loss = 0

        for i, (positions, actions, images) in enumerate(dataLoader_validation):
            images = images.view(-1, 3 * 128 * 128)
            actions = actions.view(-1, 1)
            inputs = torch.cat((images, actions), dim=1)

            outputs = model(inputs)
            loss = criterion(outputs, positions)
            validation_loss += loss.item()

        validation_loss_graph.append(validation_loss / len(dataLoader_validation))

        print(f"Epoch: {epoch}, Training Loss: {training_loss / len(dataLoader_training)}, Validation Loss: {validation_loss / len(dataLoader_validation)}")


    # draw the loss graph
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(training_loss_graph, label="Training Loss")
    plt.plot(validation_loss_graph, label="Validation Loss")
    plt.legend()
    plt.show()

    model.train(False)

    # Model Testing

    with torch.no_grad():
        for i, (positions, actions, images) in enumerate(dataLoader_testing):
            images = images.view(-1, 3 * 128 * 128)
            actions = actions.view(-1, 1)
            inputs = torch.cat((images, actions), dim=1)

            outputs = model(inputs)

            for i in range(4):
                print(f"Predicted: {outputs[i]}, Real: {positions[i]}, difference: {outputs[i] - positions[i]}")

            # graph the predicted and real x,y positions as pairs for every single predicted-real pair
            # make the graph from -1, 1 for x and y regardless of the real positions
            plt.figure()
            for i in range(len(outputs)):
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                plt.plot(outputs[i][0], outputs[i][1], 'ro')
                plt.plot(positions[i][0], positions[i][1], 'bo')
                plt.show()





