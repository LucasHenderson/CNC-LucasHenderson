import matplotlib.pyplot as plt
import numpy as np
import random
import math

class GeneticAlgorithm:
    def __init__(self, TAM_POP, recortes_disponiveis, sheet_width, sheet_height, numero_geracoes=100):
        print("Algoritmo Genético para Otimização do Corte de Chapa. Executado por Lucas.")
        self.TAM_POP = TAM_POP
        self.initial_layout = recortes_disponiveis  # Available cut parts
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.POP = []
        self.POP_AUX = []
        self.aptidao = []
        self.numero_geracoes = numero_geracoes
        self.initialize_population()
        self.melhor_aptidoes = []
        self.optimized_layout = None  # To be set after optimization

    def initialize_population(self):
        self.POP = [random.sample(self.initial_layout, len(self.initial_layout)) for _ in range(self.TAM_POP)]
        for individuo in self.POP:
            for item in individuo:
                if "rotacao" in item:
                    item["rotacao"] = random.choice([0, 90, 180, 270])

    def evaluate(self):
        self.aptidao = []
        for individuo in self.POP:
            area_utilizada = sum(self.calculate_area(r) for r in individuo)
            aptidao = self.sheet_width * self.sheet_height - area_utilizada
            self.aptidao.append(aptidao if aptidao >= 0 else float('-inf'))

    def genetic_operators(self):
        nova_populacao = []
        for _ in range(self.TAM_POP // 2):
            pais = random.sample(self.POP, 2)
            corte = random.randint(1, len(self.initial_layout) - 1)
            filho1 = pais[0][:corte] + pais[1][corte:]
            filho2 = pais[1][:corte] + pais[0][corte:]
            if random.random() < 0.1:
                idx1, idx2 = random.sample(range(len(filho1)), 2)
                filho1[idx1], filho1[idx2] = filho1[idx2], filho1[idx1]
            if random.random() < 0.1:
                idx1, idx2 = random.sample(range(len(filho2)), 2)
                filho2[idx1], filho2[idx2] = filho2[idx2], filho2[idx1]
            nova_populacao.extend([filho1, filho2])
        self.POP = nova_populacao

    def run(self):
        for _ in range(self.numero_geracoes):
            self.evaluate()
            self.genetic_operators()
        melhor_idx = np.argmax(self.aptidao)
        self.optimized_layout = self.POP[melhor_idx]
        return self.optimized_layout

    def calculate_area(self, item):
        if item["tipo"] == "retangular":
            return item["largura"] * item["altura"]
        elif item["tipo"] == "circular":
            return math.pi * (item["r"] ** 2)
        elif item["tipo"] == "triangular":
            return (item["b"] * item["h"]) / 2
        elif item["tipo"] == "diamante":
            return (item["largura"] * item["altura"]) / 2
        return 0

    def display_layout(self, layout, title="Layout"):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.sheet_width)
        ax.set_ylim(0, self.sheet_height)
        ax.set_title(title)
        for item in layout:
            x, y = item.get("x", 0), item.get("y", 0)
            rotacao = item.get("rotacao", 0)
            if item["tipo"] == "retangular":
                largura, altura = item["largura"], item["altura"]
                if rotacao in [90, 270]:
                    largura, altura = altura, largura
                rect = plt.Rectangle((x, y), largura, altura, fill=True, edgecolor='black', facecolor='gray', alpha=0.5)
                ax.add_patch(rect)
            elif item["tipo"] == "circular":
                circ = plt.Circle((x, y), item["r"], fill=True, edgecolor='black', facecolor='blue', alpha=0.5)
                ax.add_patch(circ)
            elif item["tipo"] == "triangular":
                b, h = item["b"], item["h"]
                triangle = plt.Polygon([(x, y), (x + b / 2, y + h), (x - b / 2, y + h)], fill=True, edgecolor='black', facecolor='green', alpha=0.5)
                ax.add_patch(triangle)
            elif item["tipo"] == "diamante":
                w, h = item["largura"], item["altura"]
                diamond = plt.Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)], fill=True, edgecolor='black', facecolor='red', alpha=0.5)
                ax.add_patch(diamond)
        plt.show()

    def optimize_and_display(self):
        self.display_layout(self.initial_layout, title="Initial Layout - Genetic Algorithm")
        self.optimized_layout = self.run()
        self.display_layout(self.optimized_layout, title="Optimized Layout - Genetic Algorithm")
        return self.optimized_layout
