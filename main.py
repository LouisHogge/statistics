from typing import List
import pandas as pd
import numpy as np
import math
import scipy.special as sc
from scipy.optimize import minimize
from scipy.stats import beta
from scipy.stats import chi2
from scipy.stats import chisquare
from scipy.stats import norm
from scipy.stats import t
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


FIXED_COUNTRIES = ["USA", "Belgium", "China", "Togo"]


def population(data: pd.DataFrame, ids: List[int]) -> pd.DataFrame:
    """
    Extract a population for the original dataset.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataset obtained with pandas.read_csv
    ids: List[int]
        List of ULiege ids for each group member (e.g. s167432 and s189134 -> [20167432,20189134])

    Returns
    -------
    DataFrame containing your population
    """
    pop = data.drop(FIXED_COUNTRIES).sample(146, random_state=sum(ids))
    for c in FIXED_COUNTRIES:
        pop.loc[c] = data.loc[c]
    return pop


def beta_log_likelihood(theta, *x):
    """
    Function equal to -log L(\theta;x) to be fed to scipy.optimize.minimize

    Parameters
    ----------
    theta: theta[0] is alpha and theta[1] is beta
    x: x[0] is the data
    """
    a = theta[0]
    b = theta[1]
    n = len(x[0])

    # Log-likelihood
    obj = (a - 1) * np.log(x[0]).sum() + (b - 1) * \
        np.log(1 - x[0]).sum() - n * np.log(sc.beta(a, b))
    # We want to maximize
    sense = -1

    return sense * obj


def scientific_delta(pop: pd.DataFrame) -> float:
    """

    Parameters
    ----------
    pop: pandas.DataFrame
        Dataframe containing a column 'PIB_habitant' and 'CO2_habitant'

    Returns
    -------
    Delta value computed by scientists
    """
    median_gdp = pop["PIB_habitant"].median()
    pop["Rich"] = pop.apply(lambda x: x["PIB_habitant"] >= median_gdp, axis=1)
    means = pop.groupby("Rich")['CO2_habitant'].mean()
    return means[True] - means[False]


if __name__ == '__main__':
    """
    =================================================================
                    Enregistrement fichier personnalisé
    =================================================================
    """
    # enregistrement fichier personnalisé
    file = open('data.csv')
    pop = population(pd.read_csv('data.csv', index_col='Country'),
                     [20192814, 20193340])
    # pop.to_csv('perso_data.csv')

    """
    =================================================================
                            Panneau de contrôle
    =================================================================
    """
    print("\n===============================================================")
    print("Pour exécuter le code relatif à une certaine question :")
    print("1. Ouvrir le main")
    print("2. Se rendre à la section \"Panneau de contrôle\" (+- ligne n°100)")
    print("3. Jouer avec les True/False (True par défaut)")
    print("\n(N.B.: Les graphiques ne sont pas \"show\" mais \"save\" dans le dossier où vous avez run le main.")
    print("===============================================================\n")

    # Pour exécuter le code relatif à une certaine question : jouer avec les True et
    # False

    # Q1
    Q1_b_i = True
    Q1_b_ii = True
    Q1_b_iii = True
    Q1_c = True

    # Q2
    Q2_b_d_e = True
    Q2_f = True
    Q2_bonus = True

    # Q3
    Q3_b_d = True
    Q3_e_f = True

    # Q4
    Q4_a_c_d = True

    """
    =================================================================
                                    Q1
    =================================================================
    """
    if Q1_b_i:
        print("\n===========")
        print("=  Q1_b_i =")
        print("===========\n")

        # calcul moyenne
        mean1 = pop["Top10"].mean()
        print("\nmean Top10:")
        print(mean1)
        mean2 = pop["CO2_habitant"].mean()
        print("\nmean CO2_habitant:")
        print(mean2)
        mean3 = pop["PIB_habitant"].mean()
        print("\nmean PIB_habitant:")
        print(mean3)

        # calcul standard deviation
        std1 = pop["Top10"].std()
        print("\nstandard deviation Top10:")
        print(std1)
        std2 = pop["CO2_habitant"].std()
        print("\nstandard deviation CO2_habitant:")
        print(std2)
        std3 = pop["PIB_habitant"].std()
        print("\nstandard deviation PIB_habitant:")
        print(std3)

    if Q1_b_ii:
        print("\n===========")
        print("= Q1_b_ii =")
        print("===========\n")

        # calcul median
        med1 = pop["Top10"].median()
        print("\nmedian Top10:")
        print(med1)
        med2 = pop["CO2_habitant"].median()
        print("\nmedian CO2_habitant:")
        print(med2)
        med3 = pop["PIB_habitant"].median()
        print("\nmedian PIB_habitant:")
        print(med3)

        # calcul quartiles
        q1 = pop["Top10"].quantile([.25, .75])
        print("\nquantiles Top10:")
        print(q1)
        q2 = pop["CO2_habitant"].quantile([.25, .75])
        print("\nquantiles CO2_habitant:")
        print(q2)
        q3 = pop["PIB_habitant"].quantile([.25, .75])
        print("\nquantiles PIB_habitant:")
        print(q3)

        # boxplot
        boxplot1 = pop.boxplot(column=['Top10'], grid=False)
        boxplot1.set(xticklabels=[])
        boxplot1.tick_params(bottom=False)
        plt.title('Boite à moustache de la variable Top 10')
        plt.ylabel('Revenu national détenu/Top 10 [%]')
        plt.savefig('boxplot_Top10.png', bbox_inches='tight')
        # plt.show()
        plt.clf()

        boxplot2 = pop.boxplot(column=['CO2_habitant'], grid=False)
        boxplot2.set(xticklabels=[])
        boxplot2.tick_params(bottom=False)
        plt.title('Boite à moustache de la variable CO2_habitant')
        plt.ylabel('CO2/habitant [tCO2/an]')
        plt.savefig('boxplot_CO2_habitant.png', bbox_inches='tight')
        # plt.show()
        plt.clf()

        boxplot3 = pop.boxplot(column=['PIB_habitant'], grid=False)
        boxplot3.set(xticklabels=[])
        boxplot3.tick_params(bottom=False)
        plt.title('Boite à moustache de la variable PIB_habitant')
        plt.ylabel('PIB/habitant [€]')
        plt.savefig('boxplot_PIB_habitant.png', bbox_inches='tight')
        # plt.show()
        plt.clf()

    if Q1_b_iii:
        print("\n===========")
        print("=Q1_b_iii =")
        print("===========\n")

        # histogrammes
        hist1 = pop.hist(column=['Top10'], grid=False,
                         ec='black', density=True)
        plt.title('Distribution empirique de la variable Top 10')
        plt.xlabel('Revenu national détenu/Top 10 [%]')
        plt.ylabel('Densité')
        plt.savefig('hist_Top10.png', bbox_inches='tight')
        plt.clf()

        hist2 = pop.hist(column=['CO2_habitant'],
                         grid=False, ec='black', density=True)
        plt.title('Distribution empirique de la variable CO2_habitant')
        plt.xlabel('CO2/habitant [tCO2/an]')
        plt.ylabel('Densité')
        plt.savefig('hist_CO2_habitant.png', bbox_inches='tight')
        plt.clf()

        hist3 = pop.hist(column=['PIB_habitant'],
                         grid=False, ec='black', density=True)
        plt.title('Distribution empirique de la variable PIB_habitant')
        plt.xlabel('PIB/habitant [€]')
        plt.ylabel('Densité')
        plt.savefig('hist_PIB_habitant.png', bbox_inches='tight')
        plt.clf()
        # plt.show()

        # empirical cumulative distribution function
        cdf1 = pop.hist(column=['Top10'], cumulative=True,
                        density=1, histtype='step', grid=False)
        plt.title('Fonction de répartition empirique de la variable Top 10')
        plt.xlabel('Revenu national détenu/Top 10 [%]')
        plt.ylabel('Fréquence cumulée')
        plt.savefig('cdf_Top10.png', bbox_inches='tight')
        plt.clf()

        cdf2 = pop.hist(column=['CO2_habitant'], cumulative=True,
                        density=1, grid=False, histtype='step')
        plt.title('Fonction de répartition empirique de la variable CO2_habitant')
        plt.xlabel('CO2/habitant [tCO2/an]')
        plt.ylabel('Fréquence cumulée')
        plt.savefig('cdf_CO2_habitant.png', bbox_inches='tight')
        plt.clf()

        cdf3 = pop.hist(column=['PIB_habitant'], cumulative=True,
                        density=1, grid=False, histtype='step')
        plt.title('Fonction de répartition empirique de la variable PIB_habitant')
        plt.xlabel('PIB/habitant [€]')
        plt.ylabel('Fréquence cumulée')
        plt.savefig('cdf_PIB_habitant.png', bbox_inches='tight')
        plt.clf()
        # plt.show()

    if Q1_c:
        print("\n===========")
        print("=  Q1_c   =")
        print("===========\n")

        #  nuages de points des trois couples de variables
        scat1 = pop.plot.scatter('Top10', 'CO2_habitant')
        #x_axis = scat1.axes.get_xaxis()
        # x_axis.set_visible(False)
        plt.ylabel('CO2_habitant')
        plt.xlabel('Top10')
        plt.savefig('scat_Top10_CO2_habitant.png', bbox_inches='tight')
        plt.clf()

        scat2 = pop.plot.scatter('Top10', 'PIB_habitant')
        plt.xlabel('Top10')
        plt.ylabel('PIB_habitant')
        plt.savefig('scat_Top10_PIB_habitant.png', bbox_inches='tight')
        plt.clf()

        scat3 = pop.plot.scatter('CO2_habitant', 'PIB_habitant')
        #y_axis = scat3.axes.get_yaxis()
        # y_axis.set_visible(False)
        plt.xlabel('CO2_habitant')
        plt.ylabel('PIB_habitant')
        plt.savefig('scat_CO2_PIB_habitant.png', bbox_inches='tight')
        plt.clf()
        # plt.show()

    """
    =================================================================
                                    Q2
    =================================================================
    """
    if Q2_b_d_e:
        print("\n===========")
        print("=Q2_b_d_e =")
        print("===========\n")

        # échantillon aléatoire de 50 pays
        sample_Top10 = pop["Top10"].sample(n=50)

        # calcul moyenne de l'échantillon aléatoire de 50 pays
        mean_sample_Top10 = sample_Top10.mean()

        # calcul standard deviation de l'échantillon aléatoire de 50 pays
        std_sample_Top10 = sample_Top10.std()

        # Q_2_b

        # calcul des estimateurs des  paramètres a et b en utilisant la méthode des
        # moments.
        a_MOM = mean_sample_Top10 * \
            (((mean_sample_Top10 * (1 - mean_sample_Top10)) / std_sample_Top10**2) - 1)
        print("\na_MOM:")
        print(a_MOM)

        b_MOM = ((mean_sample_Top10 - mean_sample_Top10**2 + std_sample_Top10**2)
                 * (1 - mean_sample_Top10)) / std_sample_Top10**2
        print("\nb_MOM:")
        print(b_MOM)

        # Q_2_d

        # calcul des estimateurs du maximum de vraisemblance
        ab_MLE = minimize(fun=beta_log_likelihood,
                          x0=np.array([1, 1]), args=sample_Top10)
        a_MLE = ab_MLE.x[0]
        b_MLE = ab_MLE.x[1]
        print("\na_MLE:")
        print(a_MLE)
        print("\nb_MLE:")
        print(b_MLE)

        # Q2_e

        x = np.linspace(0.2, 0.7, num=100)

        superpo_Top10 = sample_Top10.hist(
            grid=False, ec='black', density=True)
        superpo_beta_MOM = beta.pdf(x, a=a_MOM, b=b_MOM)
        plt.plot(x, superpo_beta_MOM)
        plt.title(
            'Superposition hist. échantillon pop. Top10 avec distrib. beta(a, b) MOM')
        plt.xlabel('Revenu national détenu/Top 10 [%]')
        plt.ylabel('Densité')
        plt.gca().legend(('Pdf distrib. Beta MOM', 'Distrib. empirique Top 10'))
        plt.savefig('superpo_MOM.png', bbox_inches='tight')
        plt.clf()

        superpo_Top10 = sample_Top10.hist(grid=False, ec='black', density=True)
        superpo_beta_MLE = beta.pdf(x, a=a_MLE, b=b_MLE)
        plt.plot(x, superpo_beta_MLE)
        plt.title(
            'Superposition hist. échantillon pop. Top10 avec distrib. beta(a, b) MLE')
        plt.xlabel('Revenu national détenu/Top 10 [%]')
        plt.ylabel('Densité')
        plt.gca().legend(('Pdf distrib. Beta MLE', 'Distrib. empirique Top 10'))
        plt.savefig('superpo_MLE.png', bbox_inches='tight')
        plt.clf()
        # plt.show()

    if Q2_f:
        print("\n===========")
        print("=  Q2_f   =")
        print("===========\n")

        # vraies valeurs: a = 13.35, b = 16.31
        true_a_array = np.full((500), 13.35)
        true_b_array = np.full((500), 16.31)

        a_500_MOM = np.zeros(500)
        b_500_MOM = np.zeros(500)
        a_500_MLE = np.zeros(500)
        b_500_MLE = np.zeros(500)
        for i in range(0, 500):
            # aléatoire de 50 pays
            sample_500_Top10 = pop["Top10"].sample(n=50)

            # calcul moyenne de l'échantillon aléatoire de 50 pays
            mean_sample_500_Top10 = sample_500_Top10.mean()

            # calcul standard deviation de l'échantillon aléatoire de 50 pays
            std_sample_500_Top10 = sample_500_Top10.std()

            # calcul des estimateurs des  paramètres a et b en utilisant la méthode des
            # moments.
            a_500_MOM[i] = mean_sample_500_Top10 * \
                (((mean_sample_500_Top10 * (1 - mean_sample_500_Top10)) /
                  std_sample_500_Top10**2) - 1)

            b_500_MOM[i] = ((mean_sample_500_Top10 - mean_sample_500_Top10**2 + std_sample_500_Top10**2)
                            * (1 - mean_sample_500_Top10)) / std_sample_500_Top10**2

            # calcul des estimateurs du maximum de vraisemblance
            ab_500_MLE = minimize(fun=beta_log_likelihood,
                                  x0=np.array([1, 1]), args=sample_500_Top10)
            a_500_MLE[i] = ab_500_MLE.x[0]
            b_500_MLE[i] = ab_500_MLE.x[1]

        # calcul biais estimateur MOM
        biais_a_MOM = a_500_MOM.mean() - 13.35
        biais_b_MOM = b_500_MOM.mean() - 16.31
        print("\nbiais_a_MOM:")
        print(biais_a_MOM)
        print("\nbiais_b_MOM:")
        print(biais_b_MOM)

        # calcul variance estimateur MOM
        var_a_MOM = np.var(a_500_MOM)
        var_b_MOM = np.var(b_500_MOM)
        print("\nvar_a_MOM:")
        print(var_a_MOM)
        print("\nvar_b_MOM:")
        print(var_b_MOM)

        # calcul erreur quadratique moyenne estimateur MOM
        mse_a_MOM = (np.square(np.subtract(a_500_MOM, true_a_array))).mean()
        mse_b_MOM = (np.square(np.subtract(b_500_MOM, true_b_array))).mean()
        print("\nmse_a_MOM:")
        print(mse_a_MOM)
        print("\nmse_b_MOM:")
        print(mse_b_MOM)

        # calcul biais estimateur MLE
        biais_a_MLE = a_500_MLE.mean() - 13.35
        biais_b_MLE = b_500_MLE.mean() - 16.31
        print("\nbiais_a_MLE:")
        print(biais_a_MLE)
        print("\nbiais_b_MLE:")
        print(biais_b_MLE)

        # calcul variance estimateur MLE
        var_a_MLE = np.var(a_500_MLE)
        var_b_MLE = np.var(b_500_MLE)
        print("\nvar_a_MLE:")
        print(var_a_MLE)
        print("\nvar_b_MLE:")
        print(var_b_MLE)

        # calcul erreur quadratique moyenne estimateur MLE
        mse_a_MLE = (np.square(np.subtract(a_500_MLE, true_a_array))).mean()
        mse_b_MLE = (np.square(np.subtract(b_500_MLE, true_b_array))).mean()
        print("\nmse_a_MLE:")
        print(mse_a_MLE)
        print("\nmse_b_MLE:")
        print(mse_b_MLE)

    if Q2_bonus:
        print("\n===========")
        print("=Q2_bonus =")
        print("===========\n")

        for bonus in range(20, 101, 20):
            print("##########\nbonus : " + str(bonus))
            print("##########")

            # vraies valeurs: a = 13.35, b = 16.31
            true_a_array = np.full((500), 13.35)
            true_b_array = np.full((500), 16.31)

            a_500_MOM = np.zeros(500)
            b_500_MOM = np.zeros(500)
            a_500_MLE = np.zeros(500)
            b_500_MLE = np.zeros(500)
            for i in range(0, 500):
                # échantillons aléatoire de "bonus" pays
                sample_500_bonus = pop["Top10"].sample(n=bonus)

                # calcul moyenne de l'échantillon aléatoire de "bonus" pays
                mean_sample_500_bonus = sample_500_bonus.mean()

                # calcul standard deviation de l'échantillon aléatoire de "bonus" pays
                std_sample_500_bonus = sample_500_bonus.std()

                # calcul des estimateurs des  paramètres a et b en utilisant la méthode des
                # moments.
                a_500_MOM[i] = mean_sample_500_bonus * \
                    (((mean_sample_500_bonus * (1 - mean_sample_500_bonus)) /
                      std_sample_500_bonus**2) - 1)

                b_500_MOM[i] = ((mean_sample_500_bonus - mean_sample_500_bonus**2 +
                                 std_sample_500_bonus**2) * (1 - mean_sample_500_bonus)) / std_sample_500_bonus**2

                # calcul des estimateurs du maximum de vraisemblance
                ab_500_MLE = minimize(fun=beta_log_likelihood, x0=[
                    1, 1], args=sample_500_bonus)
                a_500_MLE[i] = ab_500_MLE.x[0]
                b_500_MLE[i] = ab_500_MLE.x[1]

            # calcul biais estimateur MOM
            biais_a_MOM = a_500_MOM.mean() - 13.35
            biais_b_MOM = b_500_MOM.mean() - 16.31
            print("\nbiais_a_MOM:")
            print(biais_a_MOM)
            print("\nbiais_b_MOM:")
            print(biais_b_MOM)

            # calcul var estimateur MOM
            var_a_MOM = np.var(a_500_MOM)
            var_b_MOM = np.var(b_500_MOM)
            print("\nvar_a_MOM:")
            print(var_a_MOM)
            print("\nvar_b_MOM:")
            print(var_b_MOM)

            # calcul erreur quadratique moyenne estimateur MOM
            mse_a_MOM = (np.square(np.subtract(
                a_500_MOM, true_a_array))).mean()
            mse_b_MOM = (np.square(np.subtract(
                b_500_MOM, true_b_array))).mean()
            print("\nmse_a_MOM:")
            print(mse_a_MOM)
            print("\nmse_b_MOM:")
            print(mse_b_MOM)

            # calcul biais estimateur MLE
            biais_a_MLE = a_500_MLE.mean() - 13.35
            biais_b_MLE = b_500_MLE.mean() - 16.31
            print("\nbiais_a_MLE:")
            print(biais_a_MLE)
            print("\nbiais_b_MLE:")
            print(biais_b_MLE)

            # calcul var estimateur MLE
            var_a_MLE = np.var(a_500_MLE)
            var_b_MLE = np.var(b_500_MLE)
            print("\nvar_a_MLE:")
            print(var_a_MLE)
            print("\nvar_b_MLE:")
            print(var_b_MLE)

            # calcul erreur quadratique moyenne estimateur MLE
            mse_a_MLE = (np.square(np.subtract(
                a_500_MLE, true_a_array))).mean()
            mse_b_MLE = (np.square(np.subtract(
                b_500_MLE, true_b_array))).mean()
            print("\nmse_a_MLE:")
            print(mse_a_MLE)
            print("\nmse_b_MLE:")
            print(mse_b_MLE)
            print("\n")

    """
    =================================================================
                                    Q3
    =================================================================
    """
    if Q3_b_d:
        print("\n===========")
        print("=  Q3_b_d =")
        print("===========\n")

        # échantillon aléatoire de 50 pays
        sample_PIB_habitant = pop["PIB_habitant"].sample(n=50)

        # Q3_b méthode pivot

        # somme des PIB de l'échantillon aléatoire
        somme_PIB_habitant = 0
        for i in range(50):
            somme_PIB_habitant += sample_PIB_habitant[i]

        # calcul borne min intervalle
        intervalle_min_pivot = chi2.ppf(
            q=0.05/2, df=2*50)/(2*somme_PIB_habitant)
        print("intervalle_min_pivot:")
        print(intervalle_min_pivot)

        # calcul borne max intervalle
        intervalle_max_pivot = chi2.ppf(
            q=1.0-(0.05/2), df=2*50)/(2*somme_PIB_habitant)
        print("\nintervalle_max_pivot:")
        print(intervalle_max_pivot)

        # Q3_d méthode boostrap (méthode 1)

        # calcul 100 échantillons bootstrap
        lambda_bootstrap = np.zeros(100)
        for i in range(0, 100):

            # échantillon aléatoire de 50 pays
            sample_PIB_habitant_bootstrap = pop["PIB_habitant"].sample(
                n=50)

            # somme des PIB/habitant échantillon aléatoire de 50 pays
            somme_PIB_habitant_bootstrap = 0
            for j in range(50):
                somme_PIB_habitant_bootstrap += sample_PIB_habitant_bootstrap[j]

            # calcul des lambda par échantillon bootstrap
            lambda_bootstrap[i] = 50 / somme_PIB_habitant_bootstrap  # MLE expo

        # calcul v_boot méthode 1 bootstrap
        somme_i = 0
        for i in range(0, 100):
            somme_j = 0
            for j in range(0, 100):
                somme_j += lambda_bootstrap[j] / 100
            somme_i += (lambda_bootstrap[i] - somme_j)**2
        v_boot = somme_i / 99

        # calcul du lambda de l'échantillon commun avec méthode pivot
        lambda_bootstrap_init = 50 / somme_PIB_habitant  # MLE expo

        # calcul borne min intervalle
        intervalle_min_bootstrap = lambda_bootstrap_init - \
            (norm.ppf(1.0-0.05/2) * math.sqrt(v_boot))
        print("\nintervalle_min_bootstrap:")
        print(intervalle_min_bootstrap)

        # calcul borne max intervalle
        intervalle_max_bootstrap = lambda_bootstrap_init + \
            (norm.ppf(1.0-0.05/2) * math.sqrt(v_boot))
        print("\nintervalle_max_bootstrap:")
        print(intervalle_max_bootstrap)

    if Q3_e_f:
        print("\n===========")
        print("= Q3_e_f  =")
        print("===========\n")

        # tailles échantillons pivot
        largeur_pivot_5 = np.zeros(500)
        largeur_pivot_10 = np.zeros(500)
        largeur_pivot_15 = np.zeros(500)
        largeur_pivot_20 = np.zeros(500)
        largeur_pivot_25 = np.zeros(500)
        largeur_pivot_30 = np.zeros(500)
        largeur_pivot_35 = np.zeros(500)
        largeur_pivot_40 = np.zeros(500)
        largeur_pivot_45 = np.zeros(500)
        largeur_pivot_50 = np.zeros(500)

        # tailles échantillons bootstrap
        largeur_bootstrap_5 = np.zeros(500)
        largeur_bootstrap_10 = np.zeros(500)
        largeur_bootstrap_15 = np.zeros(500)
        largeur_bootstrap_20 = np.zeros(500)
        largeur_bootstrap_25 = np.zeros(500)
        largeur_bootstrap_30 = np.zeros(500)
        largeur_bootstrap_35 = np.zeros(500)
        largeur_bootstrap_40 = np.zeros(500)
        largeur_bootstrap_45 = np.zeros(500)
        largeur_bootstrap_50 = np.zeros(500)

        # Proportion d'intervalles contenant la vraie valeur de λ
        proportion_pivot = np.zeros(10)
        proportion_bootstrap = np.zeros(10)

        # calcul des échantillons de différentes tailles
        for taille in range(5, 51, 5):
            print("\n###########")
            print("taille: " + str(taille))
            print("###########\n")

            # 500 échantillons
            for i in range(0, 500):

                # échantillon aléatoire de "taille" pays
                sample_500_PIB_habitant = pop["PIB_habitant"].sample(n=taille)

                # méthode pivot

                # somme des PIB/habitant
                somme_500_PIB_habitant = 0
                for j in range(taille):
                    somme_500_PIB_habitant += sample_500_PIB_habitant[j]

                # borne min intervalle
                intervalle_500_min_pivot = chi2.ppf(
                    q=0.05/2, df=2*taille)/(2*somme_500_PIB_habitant)

                # borne max intervalle
                intervalle_500_max_pivot = chi2.ppf(
                    q=1.0-(0.05/2), df=2*taille)/(2*somme_500_PIB_habitant)

                # méthode bootstrap (méthode 1)

                # calcul 100 échantillons bootstrap
                lambda_bootstrap_500 = np.zeros(100)
                for j in range(0, 100):

                    # échantillon aléatoire de "taille" pays
                    sample_PIB_habitant_bootstrap_500 = pop["PIB_habitant"].sample(
                        n=taille)

                    # somme des PIB/habitant
                    somme_PIB_habitant_bootstrap_500 = 0
                    for k in range(taille):
                        somme_PIB_habitant_bootstrap_500 += sample_PIB_habitant_bootstrap_500[k]

                    # calcul des lambda par échantillon bootstrap
                    lambda_bootstrap_500[j] = 50 / \
                        somme_PIB_habitant_bootstrap_500  # MLE expo

                # calcul v_boot méthode 1 bootstrap
                somme_j_500 = 0
                for j in range(0, 100):
                    somme_k_500 = 0
                    for k in range(0, 100):
                        somme_k_500 += lambda_bootstrap_500[k] / 100
                    somme_j_500 += (lambda_bootstrap_500[j] - somme_k_500)**2
                v_boot_500 = somme_j_500 / 99

                # calcul du lambda de l'échantillon commun avec méthode pivot
                lambda_bootstrap_init_500 = 50 / somme_500_PIB_habitant

                # borne min intervalle
                intervalle_500_min_bootstrap = lambda_bootstrap_init_500 - \
                    (norm.ppf(1.0-0.05/2) * math.sqrt(v_boot_500))

                # borne max intervalle
                intervalle_500_max_bootstrap = lambda_bootstrap_init_500 + \
                    (norm.ppf(1.0-0.05/2) * math.sqrt(v_boot_500))

                # valeur vrai lambda = 5.247*10^-5
                vrai_lambda = 5.247 * 10 ** -5

                # récolte des données par taille d'échantillon
                if taille == 5:
                    largeur_pivot_5[i] = intervalle_500_max_pivot - \
                        intervalle_500_min_pivot
                    largeur_bootstrap_5[i] = intervalle_500_max_bootstrap - \
                        intervalle_500_min_bootstrap

                    if intervalle_500_min_pivot <= vrai_lambda and vrai_lambda <= intervalle_500_max_pivot:
                        proportion_pivot[0] += 100/500

                    if intervalle_500_min_bootstrap <= vrai_lambda and vrai_lambda <= intervalle_500_max_bootstrap:
                        proportion_bootstrap[0] += 100/500

                elif taille == 10:
                    largeur_pivot_10[i] = intervalle_500_max_pivot - \
                        intervalle_500_min_pivot
                    largeur_bootstrap_10[i] = intervalle_500_max_bootstrap - \
                        intervalle_500_min_bootstrap

                    if intervalle_500_min_pivot <= vrai_lambda and vrai_lambda <= intervalle_500_max_pivot:
                        proportion_pivot[1] += 100/500

                    if intervalle_500_min_bootstrap <= vrai_lambda and vrai_lambda <= intervalle_500_max_bootstrap:
                        proportion_bootstrap[1] += 100/500

                elif taille == 15:
                    largeur_pivot_15[i] = intervalle_500_max_pivot - \
                        intervalle_500_min_pivot
                    largeur_bootstrap_15[i] = intervalle_500_max_bootstrap - \
                        intervalle_500_min_bootstrap

                    if intervalle_500_min_pivot <= vrai_lambda and vrai_lambda <= intervalle_500_max_pivot:
                        proportion_pivot[2] += 100/500

                    if intervalle_500_min_bootstrap <= vrai_lambda and vrai_lambda <= intervalle_500_max_bootstrap:
                        proportion_bootstrap[2] += 100/500

                elif taille == 20:
                    largeur_pivot_20[i] = intervalle_500_max_pivot - \
                        intervalle_500_min_pivot
                    largeur_bootstrap_20[i] = intervalle_500_max_bootstrap - \
                        intervalle_500_min_bootstrap

                    if intervalle_500_min_pivot <= vrai_lambda and vrai_lambda <= intervalle_500_max_pivot:
                        proportion_pivot[3] += 100/500

                    if intervalle_500_min_bootstrap <= vrai_lambda and vrai_lambda <= intervalle_500_max_bootstrap:
                        proportion_bootstrap[3] += 100/500

                elif taille == 25:
                    largeur_pivot_25[i] = intervalle_500_max_pivot - \
                        intervalle_500_min_pivot
                    largeur_bootstrap_25[i] = intervalle_500_max_bootstrap - \
                        intervalle_500_min_bootstrap

                    if intervalle_500_min_pivot <= vrai_lambda and vrai_lambda <= intervalle_500_max_pivot:
                        proportion_pivot[4] += 100/500

                    if intervalle_500_min_bootstrap <= vrai_lambda and vrai_lambda <= intervalle_500_max_bootstrap:
                        proportion_bootstrap[4] += 100/500

                elif taille == 30:
                    largeur_pivot_30[i] = intervalle_500_max_pivot - \
                        intervalle_500_min_pivot
                    largeur_bootstrap_30[i] = intervalle_500_max_bootstrap - \
                        intervalle_500_min_bootstrap

                    if intervalle_500_min_pivot <= vrai_lambda and vrai_lambda <= intervalle_500_max_pivot:
                        proportion_pivot[5] += 100/500

                    if intervalle_500_min_bootstrap <= vrai_lambda and vrai_lambda <= intervalle_500_max_bootstrap:
                        proportion_bootstrap[5] += 100/500

                elif taille == 35:
                    largeur_pivot_35[i] = intervalle_500_max_pivot - \
                        intervalle_500_min_pivot
                    largeur_bootstrap_35[i] = intervalle_500_max_bootstrap - \
                        intervalle_500_min_bootstrap

                    if intervalle_500_min_pivot <= vrai_lambda and vrai_lambda <= intervalle_500_max_pivot:
                        proportion_pivot[6] += 100/500

                    if intervalle_500_min_bootstrap <= vrai_lambda and vrai_lambda <= intervalle_500_max_bootstrap:
                        proportion_bootstrap[6] += 100/500

                elif taille == 40:
                    largeur_pivot_40[i] = intervalle_500_max_pivot - \
                        intervalle_500_min_pivot
                    largeur_bootstrap_40[i] = intervalle_500_max_bootstrap - \
                        intervalle_500_min_bootstrap

                    if intervalle_500_min_pivot <= vrai_lambda and vrai_lambda <= intervalle_500_max_pivot:
                        proportion_pivot[7] += 100/500

                    if intervalle_500_min_bootstrap <= vrai_lambda and vrai_lambda <= intervalle_500_max_bootstrap:
                        proportion_bootstrap[7] += 100/500

                elif taille == 45:
                    largeur_pivot_45[i] = intervalle_500_max_pivot - \
                        intervalle_500_min_pivot
                    largeur_bootstrap_45[i] = intervalle_500_max_bootstrap - \
                        intervalle_500_min_bootstrap

                    if intervalle_500_min_pivot <= vrai_lambda and vrai_lambda <= intervalle_500_max_pivot:
                        proportion_pivot[8] += 100/500

                    if intervalle_500_min_bootstrap <= vrai_lambda and vrai_lambda <= intervalle_500_max_bootstrap:
                        proportion_bootstrap[8] += 100/500

                else:
                    largeur_pivot_50[i] = intervalle_500_max_pivot - \
                        intervalle_500_min_pivot
                    largeur_bootstrap_50[i] = intervalle_500_max_bootstrap - \
                        intervalle_500_min_bootstrap

                    if intervalle_500_min_pivot <= vrai_lambda and vrai_lambda <= intervalle_500_max_pivot:
                        proportion_pivot[9] += 100/500

                    if intervalle_500_min_bootstrap <= vrai_lambda and vrai_lambda <= intervalle_500_max_bootstrap:
                        proportion_bootstrap[9] += 100/500

        # plot largeurs moyennes/proportions rejet : axe x = tailles des échantillons
        taille_échantillon = np.zeros(10)
        for i in range(0, 10):
            taille_échantillon[i] = i*5

        # Q3_e

        # largeur moyenne pivot : axe y
        mean_largeur_pivot = np.zeros(10)
        mean_largeur_pivot[0] = largeur_pivot_5.mean()
        mean_largeur_pivot[1] = largeur_pivot_10.mean()
        mean_largeur_pivot[2] = largeur_pivot_15.mean()
        mean_largeur_pivot[3] = largeur_pivot_20.mean()
        mean_largeur_pivot[4] = largeur_pivot_25.mean()
        mean_largeur_pivot[5] = largeur_pivot_30.mean()
        mean_largeur_pivot[6] = largeur_pivot_35.mean()
        mean_largeur_pivot[7] = largeur_pivot_40.mean()
        mean_largeur_pivot[8] = largeur_pivot_45.mean()
        mean_largeur_pivot[9] = largeur_pivot_50.mean()

        # largeur moyenne bootstrap : axe y
        mean_largeur_bootstrap = np.zeros(10)
        mean_largeur_bootstrap[0] = largeur_bootstrap_5.mean()
        mean_largeur_bootstrap[1] = largeur_bootstrap_10.mean()
        mean_largeur_bootstrap[2] = largeur_bootstrap_15.mean()
        mean_largeur_bootstrap[3] = largeur_bootstrap_20.mean()
        mean_largeur_bootstrap[4] = largeur_bootstrap_25.mean()
        mean_largeur_bootstrap[5] = largeur_bootstrap_30.mean()
        mean_largeur_bootstrap[6] = largeur_bootstrap_35.mean()
        mean_largeur_bootstrap[7] = largeur_bootstrap_40.mean()
        mean_largeur_bootstrap[8] = largeur_bootstrap_45.mean()
        mean_largeur_bootstrap[9] = largeur_bootstrap_50.mean()

        # plot

        # gestion affichage axe x largeurs moyennes/proportions rejet
        incréments = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        localisation = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

        # Largeur moyenne intervalles en fonction de taille échantillons via méthode pivot
        plt.plot(taille_échantillon, mean_largeur_pivot, marker='o')
        plt.xlabel('Taille des échantillons [-]')
        plt.ylabel('Largeur moyenne des intervalles [-]')
        plt.title(
            'Largeur moyenne intervalles en fonction de taille échantillons via méthode pivot')
        plt.xticks(ticks=localisation, labels=incréments)
        plt.savefig('largeur_pivot.png', bbox_inches='tight')
        plt.clf()

        # Largeur moyenne intervalles en fonction de taille échantillons via méthode bootstrap
        plt.plot(taille_échantillon, mean_largeur_bootstrap, marker='o')
        plt.xlabel('Taille des échantillons [-]')
        plt.ylabel('Largeur moyenne des intervalles [-]')
        plt.title(
            'Largeur moyenne intervalles en fonction de taille échantillons via méthode bootstrap')
        plt.xticks(ticks=localisation, labels=incréments)
        plt.savefig('largeur_bootstrap.png', bbox_inches='tight')
        plt.clf()

        # Q3_f

        # plot

        # Propor. interv. contenant vraie valeur λ en fonction de taille échantillons via méthode pivot
        plt.plot(taille_échantillon, proportion_pivot, marker='o')
        plt.xlabel('Taille des échantillons [-]')
        plt.ylabel(
            'Proportion d\'intervalles contenant vraie valeur λ [%]')
        plt.title(
            'Propor. interv. contenant vraie valeur λ en fonction de taille échantillons via méthode pivot')
        plt.xticks(ticks=localisation, labels=incréments)
        plt.savefig('proportion_pivot.png', bbox_inches='tight')
        plt.clf()

        # Propor. interv. contenant vraie valeur λ en fonction de taille échantillons via méthode bootstrap
        plt.plot(taille_échantillon, proportion_bootstrap, marker='o')
        plt.xlabel('Taille des échantillons [-]')
        plt.ylabel(
            'Proportion d\'intervalles contenant vraie valeur λ [%]')
        plt.title(
            'Propor. interv. contenant vraie valeur λ en fonction de taille échantillons via méthode bootstrap')
        plt.xticks(ticks=localisation, labels=incréments)
        plt.savefig('proportion_bootstrap.png', bbox_inches='tight')
        plt.clf()

    """
    =================================================================
                                    Q4
    =================================================================
    """
    if Q4_a_c_d:
        print("\n===========")
        print("=Q4_a_c_d =")
        print("===========\n")

        # Q_4_a

        # séparation pays riches-pauvres
        median_PIB_habitant = pop["PIB_habitant"].median()
        pays_riches = []
        pays_pauvres = []
        for PIB in pop["PIB_habitant"]:
            if PIB >= median_PIB_habitant:
                for colonne_index in pop.index:
                    if PIB == pop.loc[colonne_index, "PIB_habitant"]:
                        CO2 = pop.loc[colonne_index, "CO2_habitant"]
                        pays_riches.append(CO2)
            else:
                for colonne_index in pop.index:
                    if PIB == pop.loc[colonne_index, "PIB_habitant"]:
                        CO2 = pop.loc[colonne_index, "CO2_habitant"]
                        pays_pauvres.append(CO2)

        # conversion list -> dataframe
        data_pays_riches = pd.DataFrame(pays_riches, columns=['CO2_habitant'])
        data_pays_pauvres = pd.DataFrame(
            pays_pauvres, columns=['CO2_habitant'])

        # calcul moyennes CO2_habitant
        mean_data_pays_riches = data_pays_riches.mean()
        mean_data_pays_pauvres = data_pays_pauvres.mean()

        # calcul delta
        delta = scientific_delta(pop)

        # vérification hypothèse
        if mean_data_pays_riches[0] - mean_data_pays_pauvres[0] == delta:
            print("\nmoyenne d\'émission de CO2 des pays riches  - moyenne d\'émission de CO2 des pays pauvres = " + str(mean_data_pays_riches[0]) + " - " +
                  str(mean_data_pays_pauvres[0]) + " = " + str(mean_data_pays_riches[0] - mean_data_pays_pauvres[0]) + " = delta")
        else:
            print("\nmoyenne d\'émission de CO2 des pays riches  - moyenne d\'émission de CO2 des pays pauvres = " +
                  str(mean_data_pays_riches[0] - mean_data_pays_pauvres[0]))
            print("\ndelta = " + str(delta))

        # Q_4_c

        # proportion rejet
        R_75 = 0

        # 100 test
        for i in range(0, 100):

            # échantillon aléatoire de 75 pays
            sample_75 = pop.sample(75)

            # séparation pays riches-pauvres
            pays_riches_75 = []
            pays_pauvres_75 = []
            for PIB in sample_75["PIB_habitant"]:
                if PIB >= median_PIB_habitant:
                    for colonne_index in sample_75.index:
                        if PIB == sample_75.loc[colonne_index, "PIB_habitant"]:
                            CO2 = sample_75.loc[colonne_index, "CO2_habitant"]
                            pays_riches_75.append(CO2)
                else:
                    for colonne_index in sample_75.index:
                        if PIB == sample_75.loc[colonne_index, "PIB_habitant"]:
                            CO2 = sample_75.loc[colonne_index, "CO2_habitant"]
                            pays_pauvres_75.append(CO2)

            # conversion list -> dataframe
            data_pays_riches_75 = pd.DataFrame(
                pays_riches_75, columns=['CO2_habitant'])
            data_pays_pauvres_75 = pd.DataFrame(
                pays_pauvres_75, columns=['CO2_habitant'])

            # calcul moyennes CO2_habitant
            mean_data_pays_riches_75 = data_pays_riches_75.mean()
            mean_data_pays_pauvres_75 = data_pays_pauvres_75.mean()

            # calcul z
            z_75 = mean_data_pays_riches_75 - mean_data_pays_pauvres_75

            # calcul n_r et n_p
            n_r_75 = len(data_pays_riches_75)
            n_p_75 = len(data_pays_pauvres_75)

            # calcul s_x^2 et s_y^2
            s_x_75 = 0
            for j in range(0, n_r_75):
                s_x_75 += (1/(n_r_75-1)) * \
                    ((data_pays_riches_75.loc[j, "CO2_habitant"] -
                      mean_data_pays_riches_75) ** 2)
            s_y_75 = 0
            for j in range(0, n_p_75):
                s_y_75 += (1/(n_p_75-1)) * \
                    ((data_pays_pauvres_75.loc[j, "CO2_habitant"] -
                      mean_data_pays_pauvres_75) ** 2)

            # calcul s
            s_75 = math.sqrt(
                ((n_r_75-1) * s_x_75 + (n_p_75-1) * s_y_75) / (n_r_75 + n_p_75 - 2))

            # calcul test
            test_75 = (z_75 - delta) / (s_75 * math.sqrt(1/n_r_75 + 1/n_p_75))

            # calcul t(n_r+n_p-2, 1-alpha)
            t_75 = t.ppf(q=1.0-0.05, df=n_r_75+n_p_75-2)

            # vérification rejet
            if test_75[0] > t_75:
                R_75 += 1

        print("\nproportion rejet sample_75 en % :")
        print(R_75)

        # Q_4_c

        # proportion rejet
        R_25 = 0

        # 100 test
        for i in range(0, 100):

            # échantillon aléatoire de 25 pays
            sample_25 = pop.sample(25)

            # séparation pays riches-pauvres
            pays_riches_25 = []
            pays_pauvres_25 = []
            for PIB in sample_25["PIB_habitant"]:
                if PIB >= median_PIB_habitant:
                    for colonne_index in sample_25.index:
                        if PIB == sample_25.loc[colonne_index, "PIB_habitant"]:
                            CO2 = sample_25.loc[colonne_index, "CO2_habitant"]
                            pays_riches_25.append(CO2)
                else:
                    for colonne_index in sample_25.index:
                        if PIB == sample_25.loc[colonne_index, "PIB_habitant"]:
                            CO2 = sample_25.loc[colonne_index, "CO2_habitant"]
                            pays_pauvres_25.append(CO2)

            # conversion list -> dataframe
            data_pays_riches_25 = pd.DataFrame(
                pays_riches_25, columns=['CO2_habitant'])
            data_pays_pauvres_25 = pd.DataFrame(
                pays_pauvres_25, columns=['CO2_habitant'])

            # calcul moyennes CO2_habitant
            mean_data_pays_riches_25 = data_pays_riches_25.mean()
            mean_data_pays_pauvres_25 = data_pays_pauvres_25.mean()

            # calcul z
            z_25 = mean_data_pays_riches_25 - mean_data_pays_pauvres_25

            # calcul n_r et n_p
            n_r_25 = len(data_pays_riches_25)
            n_p_25 = len(data_pays_pauvres_25)

            # calcul s_x^2 et s_y^2
            s_x_25 = 0
            for j in range(0, n_r_25):
                s_x_25 += (1/(n_r_25-1)) * \
                    ((data_pays_riches_25.loc[j, "CO2_habitant"] -
                      mean_data_pays_riches_25) ** 2)
            s_y_25 = 0
            for j in range(0, n_p_25):
                s_y_25 += (1/(n_p_25-1)) * \
                    ((data_pays_pauvres_25.loc[j, "CO2_habitant"] -
                      mean_data_pays_pauvres_25) ** 2)

            # calcul s
            s_25 = math.sqrt(
                ((n_r_25-1) * s_x_25 + (n_p_25-1) * s_y_25) / (n_r_25 + n_p_25 - 2))

            # calcul test
            test_25 = (z_25 - delta) / (s_25 * math.sqrt(1/n_r_25 + 1/n_p_25))

            # calcul t(n_r+n_p-2, 1-alpha)
            t_25 = t.ppf(q=1.0-0.05, df=n_r_25+n_p_25-2)

            # vérification rejet
            if test_25[0] > t_25:
                R_25 += 1

        print("\nproportion rejet sample_25 en % :")
        print(R_25)
