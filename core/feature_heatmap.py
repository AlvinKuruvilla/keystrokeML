import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from core.feature_table import CKP_SOURCE, KeystrokeFeatureTable
from core.utils import all_ids, get_user_by_platform, map_platform_id_to_initial


class FeatureHeatMap:
    def __init__(self, ckp_source: CKP_SOURCE) -> None:
        self.sournce = ckp_source

    def plot_heatmap(self, matrix, title=None, xlabel=None, ylabel=None):
        """Generate a heatmap from the provided feature matrix and optional title"""
        ax = sns.heatmap(matrix, linewidth=0.5)
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        plt.savefig(title)

    def make_matrix(
        enroll_platform_id, enroll_session_id, kit_feature_type, source: CKP_SOURCE
    ):
        rows = []
        if not 1 <= kit_feature_type <= 4:
            raise ValueError("KIT feature type must be between 1 and 4")
        matrix = []
        ids = all_ids()
        for i in tqdm(ids):
            df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)
            if df.empty:
                print(
                    f"Skipping User: {i}, platform: {map_platform_id_to_initial(enroll_platform_id)} session {enroll_session_id}"
                )
                continue
            print(
                f"User: {i}, platform: {map_platform_id_to_initial(enroll_platform_id)} session {enroll_session_id}"
            )
            table = KeystrokeFeatureTable()
            table.find_kit_from_most_common_keypairs(df, source)
            # table.find_deft_for_df(df=df)
            table.add_user_platform_session_identifiers(
                i, enroll_platform_id, enroll_session_id
            )

            row = table.as_df()
            rows.append(row)
