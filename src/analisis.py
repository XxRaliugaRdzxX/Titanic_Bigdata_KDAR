import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/home/rali36/Documents/GitHub/Titanic_Bigdata_KDAR/data/dataset.csv')

print("Número de pasajeros:", len(df))
print("Número de columnas:", len(df.columns))
print("Variables disponibles:", df.columns.tolist())
print("\nTipos de datos:\n", df.dtypes)
print("\nValores faltantes:\n", df.isnull().sum())
print("\nRegistros duplicados:", df.duplicated().sum())
print("\nEstadísticas descriptivas:\n", df.describe(include='all'))

# 1. Tratamiento de valores faltantes
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Cabin'] = df['Cabin'].fillna('Desconocido')
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 2. Creación de variables nuevas
# Variable 1: Tamaño de la familia
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Variable 2: Categorización de Edad
bins = [0, 12, 25, 59, 120]
labels = ['Niño', 'Joven', 'Adulto', 'Adulto mayor']
df['AgeCategory'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)

# Variable 3 (Bonus): Viajar solo vs Acompañado
df['IsAlone'] = np.where(df['FamilySize'] == 1, 'Solo', 'Acompañado')


plt.style.use('ggplot')

# Visualización 1: Supervivencia por Género
supervivencia_genero = df.groupby('Sex')['Survived'].mean() * 100
plt.figure(figsize=(8, 5))
plt.bar(['Mujeres', 'Hombres'], [supervivencia_genero['female'], supervivencia_genero['male']], color=['#FF9999', '#66B2FF'])
plt.title('Porcentaje de Supervivencia por Género')
plt.ylabel('Tasa de Supervivencia (%)')
plt.savefig('/home/rali36/Documents/GitHub/Titanic_Bigdata_KDAR/outputs/resultados/supervivencia_genero.png')

# Visualización 2: Supervivencia por Clase (Pclass)
supervivencia_clase = df.groupby('Pclass')['Survived'].mean() * 100
plt.figure(figsize=(8, 5))
plt.bar(['1ra Clase', '2da Clase', '3ra Clase'], supervivencia_clase.values, color=['gold', 'silver', 'peru'])
plt.title('Porcentaje de Supervivencia por Clase')
plt.ylabel('Tasa de Supervivencia (%)')
plt.savefig('/home/rali36/Documents/GitHub/Titanic_Bigdata_KDAR/outputs/resultados/supervivencia_clase.png')

# Visualización 3: Supervivencia por Grupo de Edad
supervivencia_edad = df.groupby('AgeCategory')['Survived'].mean() * 100
plt.figure(figsize=(8, 5))
plt.bar(supervivencia_edad.index.astype(str), supervivencia_edad.values, color='#99FF99')
plt.title('Porcentaje de Supervivencia por Grupo de Edad')
plt.ylabel('Tasa de Supervivencia (%)')
plt.savefig('/home/rali36/Documents/GitHub/Titanic_Bigdata_KDAR/outputs/resultados/supervivencia_edad.png')