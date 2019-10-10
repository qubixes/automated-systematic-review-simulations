import matplotlib.pyplot as plt

from asreview.simulation.analysis import Analysis


def add_WSS(WSS, ax, col, text, box_dist=0.5):
    if WSS is None:
        return

    WSS_A = WSS[0]*100
    WSS_B = WSS[1]*100
    text_pos_x = WSS_B + box_dist
    text_pos_y = (WSS_A + 2*WSS_B)/2
    plt.plot((WSS_B, WSS_B), (WSS_B, WSS_A+WSS_B), color=col)
    bbox = dict(boxstyle='round', facecolor=col, alpha=0.5)
    ax.text(text_pos_x, text_pos_y, text, color="white", bbox=bbox)


def add_RRF(RRF, ax, col, text, box_dist=0.5):
    if RRF is None:
        return
    RRF_A = RRF[0]*100
    RRF_B = RRF[1]*100
    text_pos_x = RRF_A + box_dist
    text_pos_y = RRF_B/2
    plt.plot((RRF_B, RRF_B), (0, RRF_A), color=col)
    bbox = dict(boxstyle='round', facecolor=col, alpha=0.5)
    ax.text(text_pos_x, text_pos_y, text, color="white", bbox=bbox)


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

        _, ax = plt.subplots()

        for i, data_key in enumerate(self.analyses):
            analysis = self.analyses[data_key]
            inc_found_result = analysis.get_inc_found(WSS_measures=[95, 100],
                                                      RRF_measures=[10])
            inc_found = inc_found_result.pop("data")
            col = "C"+str(i % 10)

            box_dist = inc_found[0].max()*0.03
            add_WSS(inc_found_result["WSS95"], ax, col, "WSS@95%", box_dist)
            add_WSS(inc_found_result["WSS95"], ax, col, "WSS@95%", box_dist)
            add_WSS(inc_found_result["WSS100"], ax, col, "WSS@100%", box_dist)
#             add_RFF(inc_found_result["RRF10"], ax, col, "RRF@10%", box_dist)
#             WSS100 = inc_found_result["WSS100"]
#             plt.plot((WSS95[1], WSS95[1]), (WSS95[1], WSS95[0]+WSS95[1]), color=col)
#             plt.plot((WSS100[1], WSS100[1]), (WSS100[1], WSS100[0]+WSS100[1]), color=col)
            x_inc = inc_found[0]*100
            y_inc = inc_found[1]*100
            err_inc = inc_found[2]*100
            myplot = plt.errorbar(x_inc, y_inc, err_inc, color=col)
            legend_name.append(f"{data_key}")
            legend_plt.append(myplot)

        plt.legend(legend_plt, legend_name, loc="upper left")
        symb = "%"

        plt.xlabel(f"{symb} Reviewed")
        plt.ylabel(f"< {symb} Inclusions found >")
        plt.title("Average number of inclusions found")
        plt.grid()
        plt.show()
