import pandas as pd

from models.graph_conv_model import GraphConvModel


def main():
    df = pd.read_csv("input/Lipophilicity.csv")
    smis = df["smiles"].values

    model_path = "output/sample_model.ckpt"
    model = GraphConvModel.load_from_checkpoint(model_path)
    
    regress = model.predict(smis)

    print(regress)


if __name__ == "__main__":
    main()
