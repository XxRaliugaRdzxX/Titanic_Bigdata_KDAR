import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar la base de datos
df = pd.read_csv('/home/rali36/Documents/GitHub/Titanic_Bigdata_KDAR/data/dataset.csv')

# ¡CLAVE! Guardamos una copia del estado original y creamos el entorno de limpieza
df_original = df.copy()
df_clean = df.copy()

print("Número de pasajeros:", len(df_original))
print("Número de columnas:", len(df_original.columns))
print("Variables disponibles:", df_original.columns.tolist())
print("\nTipos de datos:\n", df_original.dtypes)
print("\nValores faltantes:\n", df_original.isnull().sum())
print("\nRegistros duplicados:", df_original.duplicated().sum())
print("\nEstadísticas descriptivas:\n", df_original.describe(include='all'))

# 1. Tratamiento de valores faltantes (Aplicado enteramente a df_clean)
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
df_clean['Cabin'] = df_clean['Cabin'].fillna('Desconocido')
df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])

# 2. Creación de variables nuevas
# Variable 1: Tamaño de la familia
df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1

# Variable 2: Categorización de Edad
bins = [0, 12, 25, 59, 120]
labels = ['Niño', 'Joven', 'Adulto', 'Adulto mayor']
df_clean['AgeCategory'] = pd.cut(df_clean['Age'], bins=bins, labels=labels, include_lowest=True)

# Variable 3 (Bonus): Viajar solo vs Acompañado
df_clean['IsAlone'] = np.where(df_clean['FamilySize'] == 1, 'Solo', 'Acompañado')

# 3. Auditoría: Gráfico del antes y después de nulos
missing_before = df_original[['Age', 'Cabin', 'Embarked']].isnull().sum()
missing_after = df_clean[['Age', 'Cabin', 'Embarked']].isnull().sum()

print("Nulos antes de la limpieza:\n", missing_before)
print("\nNulos después de la limpieza:\n", missing_after)

# Verificando las nuevas variables creadas
print("\nMuestra de las nuevas columnas creadas:")
print(df_clean[['Age', 'AgeCategory', 'SibSp', 'Parch', 'FamilySize']].head(10))


# 4. Visualizaciones (Ajustadas para extraer los datos de df_clean)
plt.style.use('ggplot')

# Visualización 1: Supervivencia por Género
supervivencia_genero = df_clean.groupby('Sex')['Survived'].mean() * 100
plt.figure(figsize=(8, 5))
plt.bar(['Mujeres', 'Hombres'], [supervivencia_genero['female'], supervivencia_genero['male']], color=['#FF9999', '#66B2FF'])
plt.title('Porcentaje de Supervivencia por Género')
plt.ylabel('Tasa de Supervivencia (%)')
plt.savefig('../outputs/resultados/supervivencia_genero.png')

# Visualización 2: Supervivencia por Clase (Pclass)
supervivencia_clase = df_clean.groupby('Pclass')['Survived'].mean() * 100
plt.figure(figsize=(8, 5))
plt.bar(['1ra Clase', '2da Clase', '3ra Clase'], supervivencia_clase.values, color=['gold', 'silver', 'peru'])
plt.title('Porcentaje de Supervivencia por Clase')
plt.ylabel('Tasa de Supervivencia (%)')
plt.savefig('../outputs/resultados/supervivencia_clase.png')

# Visualización 3: Supervivencia por Grupo de Edad
supervivencia_edad = df_clean.groupby('AgeCategory')['Survived'].mean() * 100
plt.figure(figsize=(8, 5))
plt.bar(supervivencia_edad.index.astype(str), supervivencia_edad.values, color='#99FF99')
plt.title('Porcentaje de Supervivencia por Grupo de Edad')
plt.ylabel('Tasa de Supervivencia (%)')
plt.savefig('../outputs/resultados/supervivencia_edad.png')