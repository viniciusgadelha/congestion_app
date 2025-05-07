# Código otimizado para que o Vini tenha tudo fácil e simples ;)
Na pasta ``.cache`` você tem os modelos FL e os scalers que funcionam para todos os modelos.
Na pasta ``configs`` estão os modelos centralizados e isolados. Você também encontrará um arquivo ``.yaml`` com as configurações do hiperparâmetro de treinamento. No conjunto de dados estão os arquivos que são usados como bancos de dados.

**Observação**: lembre-se de que a entrada que você precisa inserir está localizada na pasta de teste do conjunto de dados.
Por fim, os resultados dos modelos serão salvos em um arquivo Excel na pasta de resultados.

Por fim, os resultados dos modelos serão salvos em um arquivo Excel na pasta de resultados.

---

A pasta short_term_load_forecasting contém arquivos python para executar os modelos.

Você só precisa usar ``test_nn.py`` e lembre-se que para executá-lo você precisa colocar no terminal:

```bash
python .\test_nn.py centralized
```
Substitua centralizado pela estrutura que você deseja executar
Você tem o ambiente necessário na pasta venv.

Qualquer coisa que precisar, é só me avisar, Vini. Espero que gostem do jeito que está até agora.