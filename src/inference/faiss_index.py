from pathlib import Path
import numpy as np
import faiss


class FaissIndex:

    def __init__(self):

        self.index = None

    def build(self, coreset):

        dimension = coreset.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(
            coreset.astype(np.float32)
        )

        print("=" * 60)
        print("FAISS Index Built")
        print("Dimension :", dimension)
        print("Vectors   :", self.index.ntotal)
        print("=" * 60)

        return self.index

    def save(self, output_path):

        faiss.write_index(
            self.index,
            str(output_path)
        )

    def load(self, input_path):

        self.index = faiss.read_index(
            str(input_path)
        )

        return self.index