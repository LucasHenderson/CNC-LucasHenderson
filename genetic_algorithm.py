import matplotlib.pyplot as plt
import numpy as np
import random
import math
import copy

class GeneticAlgorithm:
    def __init__(self, TAM_POP, recortes_disponiveis, sheet_width, sheet_height, numero_geracoes=100):
        print("Algoritmo Genético para Otimização do Corte de Chapa. Executado por Lucas.")
        self.TAM_POP = TAM_POP
        self.initial_layout = recortes_disponiveis  # Lista de recortes disponíveis
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.POP = []
        self.aptidao = []
        self.numero_geracoes = numero_geracoes
        self.melhor_aptidoes = []
        self.optimized_layout = None  # Layout otimizado (ordem dos itens)
        self.initialize_population()

    def initialize_population(self):
        # Cada indivíduo é uma permutação dos recortes disponíveis,
        # com rotação aleatória (para os itens que possuem o atributo "rotacao").
        self.POP = []
        for _ in range(self.TAM_POP):
            # Cria uma cópia profunda para não modificar o original
            individuo = copy.deepcopy(self.initial_layout)
            random.shuffle(individuo)
            for item in individuo:
                if "rotacao" in item:
                    # A rotação agora será efetivamente considerada: 0, 90, 180 ou 270 graus.
                    item["rotacao"] = random.choice([0, 90, 180, 270])
            self.POP.append(individuo)

    def get_dimensions(self, item):
        """
        Retorna as dimensões (largura, altura) do item considerando a rotação.
        Para itens retangulares e diamantes, se a rotação for 90 ou 270, troca largura e altura.
        Para circulares, utiliza o diâmetro.
        Para triangulares, utiliza os valores de base e altura (trocando se necessário).
        """
        if item["tipo"] == "retangular":
            w = item["largura"]
            h = item["altura"]
            if "rotacao" in item and item["rotacao"] in [90, 270]:
                w, h = h, w
            return w, h
        elif item["tipo"] == "circular":
            r = item["r"]
            return 2 * r, 2 * r
        elif item["tipo"] == "triangular":
            b = item["b"]
            h = item["h"]
            if "rotacao" in item and item["rotacao"] in [90, 270]:
                b, h = h, b
            return b, h
        elif item["tipo"] == "diamante":
            w = item["largura"]
            h = item["altura"]
            if "rotacao" in item and item["rotacao"] in [90, 270]:
                w, h = h, w
            return w, h
        return 0, 0

    def pack_layout(self, layout):
        """
        Realiza o "empacotamento" dos itens na chapa utilizando uma estratégia do tipo 'shelf' (prateleira),
        garantindo que os itens fiquem dispostos lado a lado sem sobreposição e sem ultrapassar os limites da chapa.
        Retorna uma tupla (itens_posicionados, largura_bloco, altura_bloco) se o empacotamento for bem-sucedido;
        caso contrário, retorna None.
        """
        placed_items = []
        current_x = 0
        current_y = 0
        current_row_height = 0
        rows = []  # Armazena (largura da linha, altura da linha)
        
        # Trabalhamos com uma cópia dos itens para não alterar o indivíduo original
        for item in layout:
            w, h = self.get_dimensions(item)
            # Se o item não couber na linha atual, inicia nova linha.
            if current_x + w > self.sheet_width:
                # Finaliza a linha atual
                rows.append((current_x, current_row_height))
                current_y += current_row_height
                if current_y + h > self.sheet_height:
                    # Se ultrapassar a altura da chapa, o empacotamento falha
                    return None
                current_x = 0
                current_row_height = 0
            # Cria uma cópia do item e atribui a posição calculada
            new_item = copy.deepcopy(item)
            new_item["x"] = current_x
            new_item["y"] = current_y
            # Armazena também as dimensões efetivas (já considerando rotação) para desenho
            new_item["packed_width"] = w
            new_item["packed_height"] = h
            placed_items.append(new_item)
            current_x += w
            current_row_height = max(current_row_height, h)
        # Finaliza a última linha
        rows.append((current_x, current_row_height))
        # Calcula as dimensões do bloco ocupado pelos itens
        block_width = max(row[0] for row in rows)
        block_height = sum(row[1] for row in rows)
        return placed_items, block_width, block_height

    def evaluate(self):
        """
        Avalia cada indivíduo realizando o empacotamento dos itens.
        Se o empacotamento for bem-sucedido (todos os itens cabem na chapa sem sobreposição),
        a aptidão é definida como a área livre remanescente (maior quanto melhor).
        Caso contrário, a aptidão é -infinito.
        """
        self.aptidao = []
        for individuo in self.POP:
            result = self.pack_layout(individuo)
            if result is None:
                fitness = -float('inf')
            else:
                _, block_width, block_height = result
                # A área livre considerada é a área da chapa menos a área do bloco ocupado pelos itens.
                fitness = (self.sheet_width * self.sheet_height) - (block_width * block_height)
            self.aptidao.append(fitness)

    def genetic_operators(self):
        """
        Realiza a seleção, crossover e mutação para gerar uma nova população.
        O crossover é feito por corte único e a mutação consiste em trocar dois itens de posição.
        """
        nova_populacao = []
        # Realiza crossover até preencher a nova população.
        while len(nova_populacao) < self.TAM_POP:
            pais = random.sample(self.POP, 2)
            corte = random.randint(1, len(self.initial_layout) - 1)
            filho1 = pais[0][:corte] + pais[1][corte:]
            filho2 = pais[1][:corte] + pais[0][corte:]
            # Mutação: troca de posição de dois itens com probabilidade 10%
            if random.random() < 0.1:
                idx1, idx2 = random.sample(range(len(filho1)), 2)
                filho1[idx1], filho1[idx2] = filho1[idx2], filho1[idx1]
            if random.random() < 0.1:
                idx1, idx2 = random.sample(range(len(filho2)), 2)
                filho2[idx1], filho2[idx2] = filho2[idx2], filho2[idx1]
            nova_populacao.extend([filho1, filho2])
        self.POP = nova_populacao[:self.TAM_POP]

    def run(self):
        """
        Executa o algoritmo genético por um número definido de gerações.
        Retorna o melhor indivíduo (ordem e rotações) que permita empacotamento válido e
        maximize a área livre contígua.
        """
        best_individual = None
        best_fitness = -float('inf')
        for _ in range(self.numero_geracoes):
            self.evaluate()
            # Guarda a melhor aptidão da geração
            gen_best = max(self.aptidao)
            self.melhor_aptidoes.append(gen_best)
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_individual = self.POP[np.argmax(self.aptidao)]
            self.genetic_operators()
        self.optimized_layout = best_individual
        return best_individual

    def display_layout(self, layout, title="Layout"):
        """
        Exibe o layout (com os itens já posicionados) utilizando matplotlib.
        Caso os itens possuam os atributos 'packed_width' e 'packed_height', estes serão usados para desenhar.
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.sheet_width)
        ax.set_ylim(0, self.sheet_height)
        ax.set_title(title)
        for item in layout:
            x = item.get("x", 0)
            y = item.get("y", 0)
            # Usa as dimensões empacotadas se disponíveis, caso contrário, calcula a partir da rotação
            if "packed_width" in item and "packed_height" in item:
                largura = item["packed_width"]
                altura = item["packed_height"]
            else:
                largura, altura = self.get_dimensions(item)
            rotacao = item.get("rotacao", 0)
            if item["tipo"] == "retangular":
                # Se a rotação for 90 ou 270 já consideramos a troca nas dimensões
                rect = plt.Rectangle((x, y), largura, altura, fill=True, edgecolor='black', facecolor='gray', alpha=0.5)
                ax.add_patch(rect)
            elif item["tipo"] == "circular":
                # Para círculo, x e y indicam a posição do canto superior esquerdo do quadrado delimitador
                circ = plt.Circle((x + largura/2, y + altura/2), largura/2, fill=True, edgecolor='black', facecolor='blue', alpha=0.5)
                ax.add_patch(circ)
            elif item["tipo"] == "triangular":
                b, h = largura, altura
                triangle = plt.Polygon([(x, y), (x + b/2, y + h), (x - b/2, y + h)], fill=True, edgecolor='black', facecolor='green', alpha=0.5)
                ax.add_patch(triangle)
            elif item["tipo"] == "diamante":
                w, h = largura, altura
                diamond = plt.Polygon([(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)], fill=True, edgecolor='black', facecolor='red', alpha=0.5)
                ax.add_patch(diamond)
        plt.gca().invert_yaxis()  # Opcional: inverte eixo y para que (0,0) fique no canto superior esquerdo
        plt.show()

    def optimize_and_display(self):
        """
        Exibe o layout inicial (aplicando o empacotamento) e depois roda o algoritmo genético para obter
        o layout otimizado. Ao final, exibe o layout otimizado (com itens posicionados de forma contígua, sem sobreposição e dentro dos limites da chapa).
        Retorna o layout otimizado.
        """
        # Exibe o layout inicial: utiliza o layout na ordem original dos recortes disponíveis
        initial_pack = self.pack_layout(self.initial_layout)
        if initial_pack is not None:
            placed_initial, _, _ = initial_pack
            self.display_layout(placed_initial, title="Initial Layout - Genetic Algorithm")
        else:
            # Se não empacotou, exibe como está
            self.display_layout(self.initial_layout, title="Initial Layout - Genetic Algorithm (Unpacked)")

        # Roda o algoritmo genético
        best_individual = self.run()
        # Tenta empacotar o melhor layout encontrado
        optimized_pack = self.pack_layout(best_individual)
        if optimized_pack is not None:
            placed_optimized, _, _ = optimized_pack
            self.optimized_layout = placed_optimized
            self.display_layout(placed_optimized, title="Optimized Layout - Genetic Algorithm")
        else:
            # Em caso improvável de não empacotar, exibe o layout como está
            self.display_layout(best_individual, title="Optimized Layout - Genetic Algorithm (Unpacked)")
        return self.optimized_layout
