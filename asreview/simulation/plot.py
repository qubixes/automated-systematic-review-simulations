import matplotlib.pyplot as plt

from asreview.simulation.analysis import Analysis


class Plot():
    def __init__(self, data_dirs):
        self.analyses = {}

        for data_dir in data_dirs:
            new_analysis = Analysis.from_dir(data_dir)
            if new_analysis is not None:
                data_key = new_analysis.key
                self.analyses[data_key] = new_analysis

    @classmethod
    def from_dirs(cls, data_dirs):
        plot_inst = Plot(data_dirs)
        if len(plot_inst.analyses) == 0:
            return None
        return plot_inst

    def plot_inc_found(self):
        """
        Plot the number of queries that turned out to be included
        in the final review.
        """
        legend_name = []
        legend_plt = []

        for i, data_key in enumerate(self.analyses):
            analysis = self.analyses[data_key]
            inc_found = analysis.get_inc_found()
            col = "C"+str(i % 10)

            myplot = plt.errorbar(*inc_found, color=col)
            legend_name.append(f"{data_key}")
            legend_plt.append(myplot)

        plt.legend(legend_plt, legend_name, loc="upper left")
        symb = "%"

        plt.xlabel(f"{symb} Queries")
        plt.ylabel(f"< {symb} Inclusions queried >")
        plt.title("Average number of inclusions found")
        plt.grid()
        plt.show()
