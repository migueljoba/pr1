import utils


def run():
    contribution = 1
    population = 9
    for f in utils.custom_range(1, 2, step=0.1):
        print('-' * 20)
        for coop in range(1, 10):
            common = utils.payoff_pgg(
                contribution=contribution,
                factor=f,
                total_coop=coop,
                population=population
            )

            payoff = common - contribution
            # si mi pago es 0.8
            # mi tolerancia debe ser 20% para mantener estrategia
            # si mi tolerancia es menor a 20%, cambio a desertor

            min_tol = round(1 - f * coop / population, 4)
            result = f"factor: {f}, coop: {coop}, common: {round(common, 4)}, payoff: {round(payoff, 4)}, tol: {max(0, min_tol)}"
            print(result)


if __name__ == '__main__':
    run()
