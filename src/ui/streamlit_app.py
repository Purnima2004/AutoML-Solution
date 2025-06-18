"""
Streamlit UI for the AutoML solution.
This file contains the complete user-interface, including
upload, training, results and prediction pages.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

# Make internal packages importable
sys.path.append(str(Path(__file__).parent.parent))

# Local modules
from config.model_configs import (
    get_all_model_configs,
    get_model_descriptions,
)
from core.data_processor import DataProcessor
from core.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoMLApp:
    """Top-level Streamlit application class."""

    # ---------- Initialisation ----------
    def __init__(self) -> None:
        self._set_page_cfg()
        self._init_session()
        self.data_processor = DataProcessor()
        self.model_trainer: ModelTrainer | None = None

    def _set_page_cfg(self) -> None:
        st.set_page_config(
            page_title="AutoML Pro",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def _init_session(self) -> None:
        defaults: Dict[str, object] = {
            "data_processed": False,
            "models_trained": False,
            "current_page": "🏠 Home",
            "processed_data": None,
            "training_results": None,
            "selected_models": [],
            "custom_params": {},
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    # ---------- Common helpers ----------
    @staticmethod
    def _metric(label: str, value: str) -> None:
        col = st.container().columns(1)[0]
        col.metric(label, value)

    # ---------- Sidebar ----------
    def _sidebar(self) -> None:
        with st.sidebar:
            st.title("🤖 AutoML Pro")
            st.markdown("---")

            pages = [
                "🏠 Home",
                "📊 Data Upload",
                "🔧 Model Training",
                "📈 Results",
                "🔮 Predict",
            ]
            choice = st.radio("Navigation", pages, key="nav")
            if choice != st.session_state.current_page:
                st.session_state.current_page = choice
                st.rerun()

            st.markdown("---")

            if st.session_state.data_processed:
                info = st.session_state.processed_data["data_analysis"]
                st.subheader("📋 Data Info")
                st.write(f"**Rows:** {info['shape'][0]:,}")
                st.write(f"**Columns:** {info['shape'][1]:,}")
                st.write(
                    f"**Target:** {st.session_state.processed_data['target_column']}"
                )

            if st.session_state.models_trained:
                st.subheader("🏆 Best Model")
                st.write(
                    f"**Model:** {st.session_state.training_results['best_model_name']}"
                )
                st.write(
                    f"**F1 Score:** {st.session_state.training_results['best_score']:.4f}"
                )

    # ---------- Page: Home ----------
    def _page_home(self) -> None:
        st.markdown(
            "<h1 style='text-align:center;'>AutoML Pro</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center;'>Build ML models without writing code</p>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        col1.success("🚀 No-Code Solution")
        col2.info("⚡ Automated Tuning")
        col3.warning("📊 Interactive Charts")

        st.markdown("---")
        if st.button("Get Started ➡️"):
            st.session_state.current_page = "📊 Data Upload"
            st.rerun()

    # ---------- Page: Data Upload ----------
    def _page_upload(self) -> None:
        st.header("📊 Data Upload & Pre-processing")
        file = st.file_uploader("Upload CSV", type="csv")

        if not file:
            st.info("Awaiting CSV file…")
            return

        temp_path = f"temp_{file.name}"
        Path(temp_path).write_bytes(file.read())
        df = pd.read_csv(temp_path)
        st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} cols")
        st.dataframe(df.head())

        target = st.selectbox(
            "Target column", df.columns, index=len(df.columns) - 1
        )
        test_size = st.slider("Test size (%)", 10, 40, 20) / 100

        if st.button("Process Data 🔄"):
            with st.spinner("Processing…"):
                processed = self.data_processor.process_data(
                    temp_path, target_column=target, test_size=test_size
                )
            st.session_state.processed_data = processed
            st.session_state.data_processed = True
            st.success("Data processed!")
            Path(temp_path).unlink(missing_ok=True)
            st.rerun()

    # ---------- Page: Model Training ----------
    def _page_train(self) -> None:
        if not st.session_state.data_processed:
            st.error("Upload and process data first!")
            return

        configs = get_all_model_configs()
        desc = get_model_descriptions()

        st.header("🔧 Model Training")
        selected: List[str] = []
        cols = st.columns(2)
        for i, (name, txt) in enumerate(desc.items()):
            with cols[i % 2]:
                if st.checkbox(name):
                    selected.append(name)
                st.caption(txt)

        if not selected:
            st.info("Select at least one model")
            return

        cv = st.slider("CV folds", 2, 10, 3)
        n_iter = st.slider("Random search iterations", 5, 50, 10)

        if st.button("Train Models 🚀"):
            with st.spinner("Training… this may take a while"):
                self.model_trainer = ModelTrainer(configs)
                res = self.model_trainer.train_multiple_models(
                    model_names=selected,
                    X_train=st.session_state.processed_data["X_train"],
                    y_train=st.session_state.processed_data["y_train"],
                    X_test=st.session_state.processed_data["X_test"],
                    y_test=st.session_state.processed_data["y_test"],
                    classes=st.session_state.processed_data["classes"],
                )
            st.session_state.training_results = res
            st.session_state.models_trained = True
            st.session_state.model_trainer = self.model_trainer
            st.success("Training complete!")
            st.rerun()

    # ---------- Page: Results ----------
    def _page_results(self) -> None:
        if not st.session_state.models_trained:
            st.error("Train models first!")
            return

        res = st.session_state.training_results
        best = res["best_model_name"]
        st.header("📈 Results")
        st.subheader(f"Best: {best} (F1 {res['best_score']:.4f})")

        for name, ev in res["evaluation_results"].items():
            st.markdown(f"### {name}")
            st.json(ev["metrics"])
            st.write("Confusion Matrix")
            st.dataframe(ev["confusion_matrix"])

    # ---------- Page: Predict ----------
    def _page_predict(self) -> None:
        if not st.session_state.models_trained:
            st.error("Train models first!")
            return

        upl = st.file_uploader("Upload CSV for prediction", type="csv")
        if not upl:
            return

        df_new = pd.read_csv(upl)
        st.write("Preview", df_new.head())

        best_name = st.session_state.training_results["best_model_name"]
        trainer: ModelTrainer = st.session_state.model_trainer
        preds = trainer.predict(best_name, df_new)
        st.download_button(
            "Download predictions CSV",
            data=pd.DataFrame({"prediction": preds}).to_csv(index=False),
            file_name="predictions.csv",
        )

    # ---------- Router ----------
    def run(self) -> None:
        self._sidebar()
        page = st.session_state.current_page
        if page.startswith("🏠"):
            self._page_home()
        elif page.startswith("📊"):
            self._page_upload()
        elif page.startswith("🔧"):
            self._page_train()
        elif page.startswith("📈"):
            self._page_results()
        elif page.startswith("🔮"):
            self._page_predict()
        else:
            st.error("Unknown page")


# ---------- Convenience launch ----------
def main() -> None:
    AutoMLApp().run()


if __name__ == "__main__":
    main()
