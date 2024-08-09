
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter



class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

        self.linear1 = nn.Dense(10, 10)
        self.layer2 = nn.SequentialCell(
            [nn.SequentialCell([nn.Dense(10, 10), nn.Dense(10, 10), nn.Dense(10, 10)])]
        )


def main():
    net = Net()

    print("origin:")
    for name, cell in net.name_cells().items():
        print(f"-> name: {name}, cell: {cell}")
    print(f"-> layer2.0.0.weight.name: {net.layer2[0][0].weight.name}")

    print("=" * 100)

    for name, cell in net.cells_and_names():
        if isinstance(cell, nn.SequentialCell):
            for child_name, child_cell in cell.cells_and_names():
                if isinstance(child_cell, nn.Dense):
                    _replace = nn.Conv2d(10, 10, 3)
                    _replace.weight.name = f"{child_cell.weight.name}_lora"
                    cell._cells[child_name] = _replace
                    print(f"=====> replace module, name: {name}, child_name: {child_name}")

    # import pdb;pdb.set_trace()

    print("=" * 100)

    print("replace result:")
    for name, cell in net.name_cells().items():
        print(f"-> name: {name}, cell: {cell}")
    print(f"-> layer2.0.0.weight.name: {net.layer2[0][0].weight.name}")

if __name__ == '__main__':
    main()

    ops.log()
    ops.ones()
    isinstance()


