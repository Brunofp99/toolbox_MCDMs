import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

def min_max_normalization(df, scale_factor=1):
  normalized_df = pd.DataFrame()

  for column in df.columns:
    min_val = df[column].min()
    max_val = df[column].max()
    
    normalized_df[column] = df[column].apply(lambda x: scale_factor * (x - min_val) / (max_val - min_val))

  return normalized_df

def is_dominated(row, dataframe):
  for i, compare_row in dataframe.iterrows():
    if all(compare_row <= row) and any(compare_row < row):
      return True
  return False

def find_pareto_frontier(df):
  pareto_df = df.copy()
  pareto_df['Dominated'] = pareto_df.apply(is_dominated, axis=1, dataframe=pareto_df)
  pareto_frontier = pareto_df[pareto_df['Dominated'] == False].drop(columns=['Dominated'])
  return pareto_frontier

def disagreement_point(df):
  pareto_df = find_pareto_frontier(df)
  max_distance = 0
  max_point = None
  
  for index, row in df.iterrows():
    min_dist_to_frontier = np.min([np.linalg.norm(row.values - pareto_point.values) for _, pareto_point in pareto_df.iterrows()])
    
    if min_dist_to_frontier > max_distance:
      max_distance = min_dist_to_frontier
      max_point = row
          
  return max_point

def compute_l2_norm(df):
  df['Norma P2'] = np.sqrt((df**2).sum(axis=1))
  min_p2 = min(df['Norma P2'])
  index = df.loc[df['Norma P2'] == min_p2].index[0]
  df.pop(df.columns[-1])
  best_p2 = df.iloc[index]
  return best_p2

def find_equality_solution(pareto_df):
  std_values = pareto_df.std(axis=1)
  min_std_idx = std_values.idxmin()
  equality_solution = pareto_df.iloc[min_std_idx]
  return equality_solution

def find_nash_solution(pareto_df):
  min_values = pareto_df.max()
  products = (pareto_df - min_values).prod(axis=1)
  nash_idx = products.idxmax()
  nash_solution = pareto_df.iloc[nash_idx]
  return nash_solution

def find_compromise_solution(pareto_df):
  pareto_df['sum'] = pareto_df.sum(axis = 1)
  min = pareto_df['sum'].min()
  index = df.loc[df['sum'] == min].index[0]
  df.pop(pareto_df.columns[-1])
  compromise_solution = pareto_df.iloc[index]
  return compromise_solution

def apply_topsis(df, is_minimization):
  data = df.to_numpy()
  normalized_data = data / np.sqrt((data**2).sum(axis=0))

  for i in range(len(is_minimization)):
    if is_minimization[i]:
      normalized_data[:, i] = 1 - normalized_data[:, i]

  ideal_positive = np.max(normalized_data, axis=0)
  ideal_negative = np.min(normalized_data, axis=0)

  distance_positive = np.sqrt(((normalized_data - ideal_positive) ** 2).sum(axis=1))
  distance_negative = np.sqrt(((normalized_data - ideal_negative) ** 2).sum(axis=1))

  similarity_score = distance_negative / (distance_positive + distance_negative)

  best_solution_index = np.argmax(similarity_score)
  
  return df.iloc[best_solution_index]


def verify_option_index(df, options):
  index = []
  for i in options:
    mask = df.isin(i).all(axis=1)
    indices = df.index[mask].tolist()
    index.append(indices)
  return index

def create_scatter(plt, solutions):
  colors = ['blue', 'yellow', 'red', 'purple', 'brown']
  names = ['P2', 'Equality', 'Nash', 'Compromise', 'TOPSIS']
  size = 90
  index = 0
  for solution in solutions:
    plt.scatter(solution[0], solution[1], s=size, color=colors[index], label=names[index])
    size -= 20
    index += 1

def create_concat_index(solutions, columns_name):
  index_solutions = []
  index = 0
  for solution in solutions:
    index_solutions.append([])
    for name in columns_name:
      index_solutions[index].append(solution[name])
    index += 1

  return index_solutions

def create_concat_solutions(solutions, columns_name):
  concat_solutions  = {}
  for name in columns_name:
    concat_solutions[name] = []
  
  for solution in solutions:
    for name in columns_name:
      concat_solutions[name].append(solution[name])

  return concat_solutions


def visualize_data(df):
  #Extrai o numero de criterios
  cols = df.shape[1]

  #Extrai os nomes das colunas de criterios
  columns_name = list(df.columns)

  #Escala do grafico
  plt.figure(figsize=(10, 6))

  #Printa as soluções ótimas
  plt.scatter(df[columns_name[0]], df[columns_name[1]], color='silver', label='Pareto Frontier', edgecolor='black')

  #Determinando ponto desacordo e utopico
  plt.scatter(100, 100, color='red', label='Disagreement Point')
  plt.scatter(0, 0, color='green', label='Utopic Point')

  #Controlador de minimização do Topsis
  is_minimization = [True] * cols

  #Soluções
  p2 = compute_l2_norm(df)
  equality = find_equality_solution(df)
  nash = find_nash_solution(df)
  compromise = find_compromise_solution(df)
  topsis = apply_topsis(df, is_minimization)

  #Verifica se são apenas dois criterios
  if cols == 2:
    SOLUTIONS = [
      [p2.iloc[0], p2.iloc[1]], 
      [equality.iloc[0], equality.iloc[1]],
      [nash.iloc[0], nash.iloc[1]],
      [compromise.iloc[0], compromise.iloc[1]],
      [topsis.iloc[0], topsis.iloc[1]]
    ]

    index = verify_option_index(df, SOLUTIONS)

    #Cria as cordenadas do grafico
    create_scatter(plt, SOLUTIONS)

    #Atribui nome a cada eixo (x, y)
    plt.xlabel(columns_name[0])
    plt.ylabel(columns_name[1])

    #Aplica a legenda correspondente
    plt.legend()

    #Aplica um titulo
    plt.title('Visualization of Solutions')

    #Aplica um grid no grafico
    plt.grid(True)

    #Mostra o grafico
    plt.show()

  #Cria as soluções para serem adicionadas no grafico de coordenadas paralelas
  SOLUTIONS_PARALLEL = create_concat_solutions([p2, equality, nash, compromise, topsis], columns_name)

  #Cria o modelo de array que a função verify_option_index espera para achar os valores
  index_solutions = create_concat_index([p2, equality, nash, compromise, topsis], columns_name)

  #Acha os valores dentro do dataFrame normalizado
  index = verify_option_index(df, index_solutions)

  return {'index':index, 'parallel': SOLUTIONS_PARALLEL}

#Transforma o excel em um DataFrame
dataFrame = pd.read_excel('./tcc.xlsx', engine='openpyxl')
#Normaliza os dados na escala de 0 - 100
df = min_max_normalization(dataFrame, scale_factor=100)

#Aplica as soluções e as demonstram em o grafico 2D caso seja dois criterios
data = visualize_data(df)

#Demonstra  dentro do dataFrame original os valores das soluções
for i in data['index']:
 print(dataFrame.loc[i])

#Cria as cordenadas paralelas
fig = px.parallel_coordinates(data['parallel'], color_continuous_scale=px.colors.sequential.Inferno)

#Printa o grafico de distribuição paralela
fig.show()
