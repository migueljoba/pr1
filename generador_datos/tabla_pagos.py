import utils


def run():
    contribution = 1
    population = 9
    for f in utils.custom_range(2.5, 5, step=0.5):
        print('-' * 20)
        for coop in range(1, 10):
            payoff = utils.payoff_pgg(
                contribution=contribution,
                factor=f,
                total_coop=coop,
                population=population
            ) - contribution

            # si mi pago es 0.8
            # mi tolerancia debe ser 20% para mantener estrategia
            # si mi tolerancia es menor a 20%, cambio a desertor

            # p > c * (1 - t /100)
            # p > c - ct/100
            # p - c > -ct/100
            min_tol = - 100 * round((payoff - contribution) / contribution, 4)

            # por formula
            min_tol2 = round(2 - f * coop / population, 4)
            result = f"factor: {f}, coop: {coop}, payoff: {round(payoff, 4)}, tol: {min_tol}, formulation: {min_tol2}"
            print(result)


if __name__ == '__main__':
    run()
