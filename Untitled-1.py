# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# %% [markdown]
# CustomerID : Identifiant unique du client.
# 
# Gender : Sexe du client (homme ou femme).
# 
# SeniorCitizen : Indique si le client est une personne âgée (1 = oui, 0 = non).
# 
# Partner : Indique si le client est en couple (Yes / No).
# 
# Dependents : Indique si le client a des personnes à charge (enfants ou autres).
# 
# Tenure : Ancienneté du client en nombre de mois depuis son abonnement.
# 
# PhoneService : Indique si le client possède un service téléphonique.
# 
# MultipleLines : Indique si le client a plusieurs lignes téléphoniques.
# 
# InternetService : Type de service internet (DSL, fibre optique ou aucun).
# 
# OnlineSecurity : Indique si le client possède un service de sécurité en ligne.
# 
# DeviceProtection : Indique si le client a une protection pour ses appareils.
# 
# TechSupport : Indique si le client a accès à une assistance technique.
# 
# StreamingTV : Indique si le client utilise un service de télévision en streaming.
# 
# StreamingMovies : Indique si le client utilise un service de films en streaming.
# 
# Contract : Type de contrat (mensuel, 1 an, 2 ans).
# 
# PaperlessBilling : Indique si la facturation est électronique (sans papier).
# 
# PaymentMethod : Méthode de paiement utilisée par le client.
# 
# MonthlyCharges : Montant payé par le client chaque mois.
# 
# TotalCharges : Montant total payé par le client depuis le début de son abonnement.
# 
# Churn : Indique si le client s’est désabonné (Yes) ou s’il est resté (No).
# 

# %%
df = pd.read_csv("telco-Customer-Churn.csv")
df.head()

# %%
df.info()

# %%
df.describe()

# %%
df.isnull().sum()

# %%
df.duplicated().sum()

# %%
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df

# %%
df['TotalCharges'].isnull().sum()

# %%
df = df.dropna()
df

# %%
df.isnull().sum()

# %%
df['Churn'].value_counts(normalize=True)

# %% [markdown]
# Nous avons plus de 73% des clients qui conservent leurs abonnement et 26% qui se desabonne.

# %% [markdown]
# 2-Pourquoi les clients quittent-ils la plateforme ?
# 
# Pour repondre a cette interrogations nous allons proceder comme suit :
#     -Analyse du profil des clients qui churn
#     -Analyse des facteurs financiers et contractuels
#     -Identification des variables les plus liées au churn

# %% [markdown]
# 2.1- Analyse du profil des clients qui churn 

# %%
counts = df['Churn'].value_counts()
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
plt.title('Répartition du Churn')
plt.show()

# %% [markdown]
# La visualisation montre la proportion de clients qui quittent la plateforme par rapport à ceux qui restent.

# %% [markdown]
# 2-2-Analyse des facteurs financiers et contractuels

# %%
contract_churn = pd.crosstab(df['Contract'], df['Churn'])
contract_churn

# %%
chi2, p , dof, expected = stats.chi2_contingency(contract_churn)
print(f'p-value: {p}, chi:{chi2}')

# %%
contract_churn.plot(kind='bar', figsize=(8,5))
plt.xlabel('Type de contrat')
plt.ylabel("Proportion")
plt.title("Distribution du Churn selon le type de contrat")
plt.tight_layout()
plt.show()

# %% [markdown]
# Les clients ayant un contrat mensuel (Month-to-month) présentent un taux de churn plus élevé que ceux ayant un contrat d’un ou deux ans.

# %%
bins = [0, 25, 50, 75, 100, 125]
labels = ['0-25', '25-50', '50-75', '75-100', '100-125']
df['MonthlyCharges_bins'] = pd.cut(df['MonthlyCharges'], bins=bins, labels=labels, right=False)
df

# %%
monthlyCharges_churn = pd.crosstab(df['MonthlyCharges_bins'], df['Churn'])
monthlyCharges_churn

# %%
chi2, p , dof, expected = stats.chi2_contingency(monthlyCharges_churn)
print(f'p-value: {p}, chi:{chi2}')

# %%
monthlyCharges_churn.plot(kind='bar', figsize=(8,5))
plt.xlabel('MonthlyCharges')
plt.ylabel('Counts')
plt.title('Distibution des Churns a travers le Mois')
plt.tight_layout()
plt.show()

# %%
bins = [
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
        55, 60, 65, 70, 75]
labels = [
    '0-5', '5-10', '10-15', '15-20', '20-25', 
    '25-30', '30-35', '35-40', '40-45', '45-50',
    '50-55', '55-60', '60-65', '65-70', '70-75'
]

df['tenure_bins'] = pd.cut(df['tenure'], bins=bins, labels=labels)
df

# %%
tenure_churn = pd.crosstab(df['tenure_bins'], df['Churn'])
tenure_churn 

# %%
chi2, p , dof, expected = stats.chi2_contingency(tenure_churn)
print(f'p-value: {p}, chi:{chi2}')

# %%
tenure_churn.plot(kind='bar', figsize=(8,5))
plt.xlabel('tenure')
plt.ylabel('counts')
plt.title("Distibution des Churns a l'anciennete du client")
plt.tight_layout()
plt.show()

# %%
internet_service = pd.crosstab(df['InternetService'], df['Churn'])
internet_service

# %%
chi2, p , dof, expected = stats.chi2_contingency(internet_service)
print(f'p-value: {p}, chi:{chi2}')

# %%
internet_service.plot(kind='bar', figsize=(8,5))
plt.xlabel("Internet service")
plt.ylabel('Proportion')
plt.title("Distribution du Churn selon le service d'internet")
plt.tight_layout()
plt.show()

# %%
monthlyCharges_intservice = pd.crosstab([df['MonthlyCharges_bins'],df['InternetService']], df['Churn'], normalize='index')
monthlyCharges_intservice

# %%
monthlyCharges_intservice.plot(kind='bar', figsize=(8,5))
plt.xlabel('Tranches de prix et Service Internet')
plt.ylabel('Proportion de clients')
plt.title('Impact du prix et du service sur le Churn')
plt.tight_layout()
plt.show()

# %%
phone_service = pd.crosstab(df['PhoneService'], df['Churn'])
phone_service

# %%
chi2, p , dof, expected = stats.chi2_contingency(phone_service)
print(f'p-value: {p}, chi:{chi2}')

# %%
phone_service.plot(kind='bar', figsize=(8,5))
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.show()

# %%
streaming_movie = pd.crosstab(df['StreamingMovies'], df['Churn'], normalize='index')
streaming_movie

# %%
streaming_movie.plot(kind='bar', figsize=(8,5))
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.show()

# %%
streaming_tv = pd.crosstab(df['StreamingTV'], df['Churn'], normalize='index')
streaming_tv

# %%
streaming_tv.plot(kind='bar', figsize=(8,5))
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.show()

# %%
paper = pd.crosstab(df['PaperlessBilling'], df['Churn'])
paper

# %%
chi2, p , dof, expected = stats.chi2_contingency(paper)
print(f'p-value: {p}, chi:{chi2}')

# %%
paper.plot(kind='bar', figsize=(8,5))
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.show()

# %%
payement_method = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index')
payement_method

# %%
paper_payement = pd.crosstab([df['PaperlessBilling'], df['PaymentMethod']], df['Churn'], normalize='index')
paper_payement

# %%
paper_payement.plot(kind='bar', figsize=(8,5))
plt.xlabel('')
plt.ylabel("")
plt.title('')
plt.show()

# %%
streaming = pd.crosstab([df['StreamingMovies'], df['StreamingTV']], df['Churn'], normalize='index')
streaming

# %%
streaming.plot(kind='bar', figsize=(8,5))
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.show()

# %%
service = pd.crosstab([df['PhoneService'], df['InternetService']], df['Churn'], normalize='index')
service

# %%
service.plot(kind='bar', figsize=(8,5))
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.show()

# %%
contract_monthlycharge = pd.crosstab([df['MonthlyCharges_bins'], df['Contract']], df['Churn'], normalize='index')
contract_monthlycharge

# %%
contract_monthlycharge.plot(kind='bar', figsize=(8,5))
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.show()

# %%
charges = pd.crosstab([df['MonthlyCharges_bins'], df['PaymentMethod']], df['Churn'], normalize='index')
charges

# %%
charges.plot(kind='bar', figsize=(8,5))
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.show()


