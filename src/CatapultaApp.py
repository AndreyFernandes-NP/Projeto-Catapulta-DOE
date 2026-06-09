class CatapultaApp:
    def __init__(self):
        self.fontes = []
        self.datasets = {}

    def listar_fontes(self):
        self.fontes = get_csv()
        return self.fontes

    def carregar_fontes(self):
        if not self.fontes:
            self.listar_fontes()

        for fonte in self.fontes:
            nome = fonte["nome"]
            self.datasets[nome] = load_csv(fonte)

        return self.datasets

    def executar(self):
        fontes = self.listar_fontes()
        print("Fontes encontradas:", [f["nome"] for f in fontes])

        for fonte in fontes:
            nome = fonte["nome"]
            try:
                df = load_csv(fonte)
                print(f"{nome} carregado com {len(df)} linhas e {len(df.columns)} colunas.")
            except Exception as e:
                print(f"Erro ao carregar {nome}: {e}")