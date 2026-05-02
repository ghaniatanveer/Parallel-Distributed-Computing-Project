"""
Generate PDC_Health_Diagnosis_Report.docx for the university project.
Run: python build_pdc_report.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


def set_document_defaults(doc: Document) -> None:
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Times New Roman"
    font.size = Pt(12)
    pf = style.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.15
    pf.space_after = Pt(6)


def add_code_block(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_together = True
    run = p.add_run(text.strip("\n"))
    run.font.name = "Consolas"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    run.font.size = Pt(9)


def add_page_number_footer(section) -> None:
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.text = ""
    run = p.add_run("Page ")
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)
    run2 = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run2._r.append(fld_begin)
    run2._r.append(instr)
    run2._r.append(fld_end)
    run2.font.name = "Times New Roman"
    run2.font.size = Pt(11)


def heading(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def para(doc: Document, text: str, bold: bool = False) -> None:
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)


def bullet_list(doc: Document, items: list[str]) -> None:
    for t in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent = Inches(0.35)
        r = p.add_run(t)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)


def add_figure(
    doc: Document,
    image_path: Path,
    caption: str,
    width_inches: float = 6.0,
) -> bool:
    """Embed a PNG if it exists; return True if added."""
    if not image_path.is_file():
        return False
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].italic = True
    cap.runs[0].font.name = "Times New Roman"
    cap.runs[0].font.size = Pt(11)
    doc.add_paragraph()
    return True


def build_report(path: Path) -> None:
    doc = Document()
    set_document_defaults(doc)

    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    add_page_number_footer(section)

    # ----- Title Page -----
    for _ in range(6):
        doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(
        "Parallel and Distributed Computing Based Multi-Task Health Diagnosis System "
        "Using 4 Body Sensors"
    )
    r.bold = True
    r.font.size = Pt(18)
    r.font.name = "Times New Roman"
    r.font.color.rgb = RGBColor(0, 0, 0)

    doc.add_paragraph()
    st = doc.add_paragraph()
    st.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = st.add_run("Course / Project Report\nParallel and Distributed Computing")
    rr.font.size = Pt(14)
    rr.font.name = "Times New Roman"

    doc.add_paragraph()
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s = sub.add_run(
        "Submitted in partial fulfillment of the requirements for the course on "
        "Parallel and Distributed Computing"
    )
    s.font.size = Pt(12)
    s.font.italic = True

    doc.add_paragraph()
    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    m = meta.add_run(
        "Submitted by:\n"
        "Muhammad Haseeb — Registration No. L1F22BSCS0190\n"
        "Ghania Tanver — Registration No. L1F22BSCS0420\n"
        "\n"
        "Supervisor: Dr Khurram Zahur Bajwa\n"
        "\n"
        "Department / Program: BS Computer Science\n"
        f"Date: {date.today().strftime('%B %d, %Y')}"
    )
    m.font.size = Pt(12)

    doc.add_page_break()

    # ----- Abstract -----
    heading(doc, "Abstract", 1)
    para(
        doc,
        "This report presents a multi-task health diagnosis system that uses readings from four "
        "body-mounted sensors (fever, cough, sneeze, weakness) together with a migraine pain score. "
        "The system trains separate machine learning models to predict patient gender, age, and "
        "probable disease from the same feature vector. A central theme of the work is the "
        "application of parallel and distributed computing concepts: parallel training of three "
        "independent models using Python's concurrent.futures.ThreadPoolExecutor, internal "
        "parallelism inside Random Forest learners via n_jobs=-1, and parallel inference for "
        "low-latency multi-output prediction. A synthetic dataset of 20,000 patient records was "
        "generated to develop and evaluate the pipeline. On a held-out test set, the gender "
        "classifier and disease classifier achieve high accuracy, while age is modeled with a "
        "regressor and evaluated using R-squared and mean absolute error. The report discusses "
        "implementation details, results, challenges, and how the design could scale to real "
        "distributed engines such as Apache Spark or Dask and to physical sensor hardware. "
        "Section 3.12 summarizes which tasks run concurrently for oral examination (viva); "
        "Section 6.4 embeds bar-chart and confusion-matrix figures saved under plot_outputs/.",
    )

    # ----- 1 Introduction -----
    heading(doc, "1. Introduction", 1)
    heading(doc, "1.1 Problem Statement", 2)
    para(
        doc,
        "Healthcare monitoring increasingly relies on multiple data streams collected at the same "
        "time. When several diagnostic questions must be answered from one set of measurements "
        "(for example demographic hints and disease likelihood), a naive sequential pipeline "
        "trains or runs one model after another. That approach increases wall-clock time and "
        "delays decisions. From a computing perspective, these tasks are often independent: "
        "predicting gender does not require the age model to finish first, and disease "
        "probabilities can be computed concurrently as long as each model receives the same "
        "input features. This project addresses that situation by structuring the software so "
        "that parallelism is explicit in training and inference, while still using a single "
        "workstation or server for development and demonstration.",
    )
    para(
        doc,
        "The system is demonstrated on a CSV dataset produced by a reproducible generator "
        "(dataset.py). The dataset encodes seven disease labels and continuous sensor scales so "
        "that learning algorithms can be validated before any real hardware is attached.",
    )

    heading(doc, "1.2 Objectives", 2)
    bullet_list(
        doc,
        [
            "Build a dataset of 20,000 records with demographics, four sensor channels, migraine "
            "pain, and disease labels.",
            "Train three models: (i) classification of gender, (ii) regression of age, "
            "(iii) classification of disease among seven classes.",
            "Apply parallel training using ThreadPoolExecutor with three worker threads, one per "
            "model, and use Random Forest's n_jobs=-1 for intra-model parallelism.",
            "Implement parallel inference that runs gender prediction, age prediction, and "
            "disease probability estimation concurrently.",
            "Evaluate models with accuracy (classification), R-squared and MAE (age), confusion "
            "matrices, and report an example Typhoid-like sensor pattern.",
            "Document how these techniques relate to broader parallel and distributed computing "
            "ideas and how they could scale with Spark, Dask, or sensor clusters.",
        ],
    )

    # ----- 2 Dataset -----
    heading(doc, "2. Dataset Description", 1)
    para(
        doc,
        "The file health_dataset_20000.csv is generated by dataset.py. The generator uses a fixed "
        "random seed (42) so that experiments are repeatable. Each row represents one patient.",
    )

    heading(doc, "2.1 Generation of 20,000 Records", 2)
    para(
        doc,
        "For each of 20,000 rows, a disease label is drawn uniformly at random from seven "
        "conditions: Typhoid, Pneumonia, Dengue, Flu, Common Cold, Malaria, and Only Fever. "
        "Conditional on the disease, five continuous features are sampled from truncated normal "
        "distributions with disease-specific means and controlled standard deviations. Values are "
        "clipped to realistic ranges (0–5 for sensors, 0–10 for migraine pain). Tighter noise "
        "helps keep classes separable for evaluation. Gender is sampled (approximately 52% Male, "
        "48% Female). Age is drawn from a normal distribution with mean 35 and standard deviation "
        "20 years, clipped to ages 1 through 95 so that children and older adults appear in the "
        "data. Small learnable shifts are applied to cough and sneeze by gender, and a mild age "
        "dependent adjustment is added across sensors so that the age regressor has a learnable "
        "signal without destroying disease separation.",
    )

    heading(doc, "2.2 Features: Four Body Sensors and Migraine", 2)
    para(
        doc,
        "Table 1 summarizes the input features used by all models. The naming reflects the "
        "intended physical placement: forehead (fever), chest (cough), nose (sneeze), body "
        "(weakness), plus migraine pain as an additional scalar.",
    )

    tbl = doc.add_table(rows=1, cols=3)
    tbl.style = "Table Grid"
    hdr = tbl.rows[0].cells
    hdr[0].text = "Feature name"
    hdr[1].text = "Role"
    hdr[2].text = "Typical scale"
    rows_data = [
        ("sensor1_fever", "Thermal / fever proxy (forehead)", "0–5"),
        ("sensor2_cough", "Cough / chest activity", "0–5"),
        ("sensor3_sneeze", "Nasal / sneeze activity", "0–5"),
        ("sensor4_weakness", "General weakness / fatigue", "0–5"),
        ("migraine_pain", "Headache / migraine intensity", "0–10"),
    ]
    for name, role, scale in rows_data:
        row = tbl.add_row().cells
        row[0].text = name
        row[1].text = role
        row[2].text = scale
    doc.add_paragraph()
    p_cap = doc.add_paragraph("Table 1: Input features for all models.")
    p_cap.runs[0].italic = True

    heading(doc, "2.3 Sample Records", 2)
    para(
        doc,
        "Table 2 shows a concise sample of columns as they appear in the CSV (values are "
        "illustrative of structure; exact numbers depend on the current seed and run).",
    )
    t2 = doc.add_table(rows=5, cols=8)
    t2.style = "Table Grid"
    sample_header = [
        "patient_id",
        "gender",
        "age",
        "sensor1_fever",
        "sensor2_cough",
        "sensor3_sneeze",
        "sensor4_weakness",
        "disease",
    ]
    for i, h in enumerate(sample_header):
        t2.rows[0].cells[i].text = h
    sample_rows = [
        ["1", "Male", "28", "2.91", "1.21", "0.45", "2.36", "Only Fever"],
        ["2", "Male", "21", "3.12", "4.05", "3.40", "3.57", "Flu"],
        ["3", "Female", "52", "1.42", "2.10", "4.18", "2.55", "Common Cold"],
        ["4", "Male", "16", "2.80", "0.95", "0.31", "1.59", "Only Fever"],
    ]
    for r_idx, row_vals in enumerate(sample_rows, start=1):
        for c_idx, val in enumerate(row_vals):
            t2.rows[r_idx].cells[c_idx].text = val
    doc.add_paragraph()
    doc.add_paragraph("Table 2: Example rows (migraine_pain column omitted for width).").runs[
        0
    ].italic = True

    # ----- 3 PDC Main Section (long) -----
    heading(doc, "3. Parallel and Distributed Computing Approach", 1)
    para(
        doc,
        "Parallel and distributed computing studies how large problems can be split into smaller "
        "tasks that run at the same time on multiple cores or multiple machines, and how such "
        "systems are coordinated, scheduled, and scaled. This project uses parallelism at two "
        "levels: (1) task-level parallelism across three independent machine learning models, and "
        "(2) data-parallel or tree-parallel work inside each Random Forest implementation. The "
        "development environment is a single Python process on one computer; however, the same "
        "patterns map naturally to distributed job queues, cluster frameworks, and microservice "
        "deployments described later.",
    )

    heading(doc, "3.1 Why Parallelism Matters for This System", 2)
    para(
        doc,
        "Multi-task diagnosis has three separate learning problems that share the same input "
        "matrix X but use different target vectors (gender encoding, age, disease). Training "
        "them strictly one after another wastes time because the gradient-free tree algorithms do "
        "not need sequential coupling. Similarly, at prediction time, three forward passes through "
        "tree ensembles are independent. Using threads to overlap CPU-bound work can reduce "
        "elapsed time when the underlying libraries release the GIL during native code execution "
        "(as many scikit-learn routines do for Random Forest fitting and prediction). Even when "
        "speedup is modest on one machine, the design mirrors production systems where separate "
        "models are deployed as separate services or workers.",
    )

    heading(doc, "3.2 ThreadPoolExecutor for Parallel Training", 2)
    para(
        doc,
        "The training phase submits three jobs to a ThreadPoolExecutor with max_workers=3. Each "
        "job calls fit on one estimator. Conceptually, this is task parallelism: three independent "
        "tasks share the same training data structure in memory but execute different instructions. "
        "The main thread waits on futures (result()) so that failures surface clearly. This "
        "pattern is easy to read and does not require manual thread management.",
    )

    heading(doc, "3.3 n_jobs=-1 and Internal Parallelism in Random Forest", 2)
    para(
        doc,
        "Each RandomForestClassifier and RandomForestRegressor is constructed with n_jobs=-1, "
        "which asks scikit-learn to use all available CPU cores for building trees in parallel. "
        "Random Forest fits many decision trees on bootstrap samples; these tree fits are a classic "
        "embarrassingly parallel workload. Thus, parallelism exists at two scales: across three "
        "models (thread pool) and inside each model (multi-core tree construction). In a "
        "distributed cluster, the outer level could map to three Spark jobs or three Dask tasks, "
        "while the inner level could use many executors per model.",
    )

    heading(doc, "3.4 Parallel Inference with ThreadPoolExecutor", 2)
    para(
        doc,
        "The function predict_parallel wraps three concurrent submissions: predict for gender, "
        "predict for age, and predict_proba for disease. Results are merged into one dictionary "
        "with human-readable gender, rounded age, top disease, and per-class disease "
        "probabilities expressed as percentages. Parallel inference reduces latency when models "
        "are large and prediction cost is non-trivial, and it demonstrates the same task "
        "parallelism idea as training.",
    )

    heading(doc, "3.5 Benefits: Speed, Efficiency, and Structure", 2)
    bullet_list(
        doc,
        [
            "Wall-clock time: overlapping three fits or three predictions can improve throughput on "
            "multi-core hosts compared to a purely sequential loop.",
            "Resource utilization: n_jobs=-1 uses available cores for tree building instead of "
            "leaving them idle while one model trains.",
            "Software clarity: separating concerns (three estimators) and using explicit executors "
            "documents intent and eases migration to process-based or distributed backends.",
        ],
    )

    heading(doc, "3.6 Relation to Distributed Computing Concepts", 2)
    para(
        doc,
        "True distributed computing typically involves multiple nodes, message passing, fault "
        "tolerance, and schedulers (for example Hadoop YARN, Kubernetes, or Spark standalone). "
        "This project does not deploy across a cluster, but it implements the same logical "
        "decomposition: independent tasks (models) that could be placed on different workers. A "
        "natural extension is to package each model as a containerized service behind a load "
        "balancer, or to train on partitions of data with MapReduce-style aggregation. Spark MLlib "
        "and Dask-ML can distribute tree methods or replace parts of the pipeline with parallel "
        "DataFrame operations. The current code therefore serves as a pedagogical bridge between "
        "single-machine concurrency and large-scale distributed ML.",
    )

    heading(doc, "3.7 Summary Table: Parallel Mechanisms", 2)
    t3 = doc.add_table(rows=1, cols=3)
    t3.style = "Table Grid"
    h3 = t3.rows[0].cells
    h3[0].text = "Mechanism"
    h3[1].text = "Where used"
    h3[2].text = "Purpose"
    for a, b, c in [
        (
            "ThreadPoolExecutor (3 workers)",
            "Training and predict_parallel",
            "Run three models concurrently",
        ),
        ("n_jobs=-1", "Each Random Forest", "Parallel tree construction / prediction"),
        ("Shared feature matrix X", "All models", "Same inputs, independent outputs"),
        ("joblib persistence", "After training", "Save models for reuse"),
    ]:
        row = t3.add_row().cells
        row[0].text = a
        row[1].text = b
        row[2].text = c
    doc.add_paragraph()
    doc.add_paragraph("Table 3: Parallel mechanisms in the implementation.").runs[0].italic = True

    heading(doc, "3.8 Workload Characterization", 2)
    para(
        doc,
        "Training Random Forest models is CPU-intensive and memory-bound. Each tree is built from "
        "a bootstrap sample of rows and a random subset of features at each split. The workload is "
        "embarrassingly parallel across trees, which is why n_jobs=-1 is effective. Across three "
        "models, the workload is embarrassingly parallel at the task level as long as RAM holds "
        "three fitted models and the shared matrix X. The project therefore exhibits both "
        "fine-grained parallelism (trees) and coarse-grained parallelism (models).",
    )

    heading(doc, "3.9 Sequential Baseline versus Parallel Design", 2)
    para(
        doc,
        "A sequential baseline would execute: fit gender; then fit age; then fit disease. Total "
        "time is approximately the sum of three training times plus sequential prediction. The "
        "parallel design replaces the training phase with three concurrent fits when threads can "
        "progress together, aiming for a shorter wall-clock time on multi-core CPUs. The exact "
        "speedup depends on hardware, BLAS threads, and whether disk I/O competes for resources. "
        "The important point for the course is architectural: the system is structured for "
        "parallelism first, which is prerequisite for scaling out to many nodes.",
    )

    heading(doc, "3.10 From Threads to Distributed Tasks", 2)
    para(
        doc,
        "In a distributed environment, each model could become a task in a directed acyclic graph "
        "(DAG) scheduler. Dask, for example, could delay three fit operations and run them on "
        "different workers if data were serialized or partitioned. Spark could distribute feature "
        "preprocessing map steps and reduce tree statistics in specialized algorithms. Message "
        "queues such as Apache Kafka could stream sensor readings to multiple inference consumers. "
        "The ThreadPoolExecutor pattern is therefore a minimal local analogue of a distributed "
        "task graph.",
    )

    heading(doc, "3.11 Python Global Interpreter Lock (GIL) Note", 2)
    para(
        doc,
        "CPython's GIL limits true parallelism for pure Python bytecode in one process. "
        "scikit-learn offloads much Random Forest work to native libraries (Cython/C) where the "
        "GIL can be released, so threads remain a practical choice for overlapping model work. "
        "If thread speedup were insufficient, multiprocessing.ProcessPoolExecutor or Ray actors "
        "could run models in separate processes with separate memory spaces, at the cost of "
        "serialization overhead for large arrays.",
    )

    heading(doc, "3.12 What Runs at the Same Time (PDC Summary for Viva)", 2)
    para(
        doc,
        "Use this subsection to justify Parallel and Distributed Computing in the oral exam: the "
        "implementation is not only \"fast ML\" but explicitly concurrent at two levels—across "
        "models and inside each Random Forest.",
    )
    t_viva = doc.add_table(rows=1, cols=2)
    t_viva.style = "Table Grid"
    hv = t_viva.rows[0].cells
    hv[0].text = "Phase / moment"
    hv[1].text = "Tasks that may execute concurrently (parallelism)"
    for a, b in [
        (
            "Training (model.py main)",
            "Three independent fits run together: (1) gender RandomForestClassifier.fit, "
            "(2) age RandomForestRegressor.fit, (3) disease RandomForestClassifier.fit—submitted "
            "to one ThreadPoolExecutor with max_workers=3, so all three training jobs overlap in "
            "wall-clock time on a multi-core CPU.",
        ),
        (
            "Inside each fit (n_jobs=-1)",
            "Each forest builds many trees from bootstrap samples; scikit-learn parallelizes tree "
            "construction across CPU cores. So while the three model-level threads are active, each "
            "model also uses data-parallel / tree-parallel work internally—fine-grained parallelism.",
        ),
        (
            "Inference (predict_parallel)",
            "Three forward passes run together: gender predict, age predict, disease predict_proba—"
            "again via ThreadPoolExecutor(max_workers=3). Outputs are merged only after all three "
            "futures complete (low-latency multi-head prediction).",
        ),
        (
            "Distributed analogy",
            "On one machine this is shared-memory concurrency; the same task graph could be "
            "mapped to separate workers or nodes (e.g. Spark/Dask/Kubernetes pods), which is the "
            "distributed computing extension of the same design.",
        ),
    ]:
        row = t_viva.add_row().cells
        row[0].text = a
        row[1].text = b
    doc.add_paragraph()
    doc.add_paragraph(
        "Table 3a: Concurrent phases—what happens at the same time for PDC justification."
    ).runs[0].italic = True
    bullet_list(
        doc,
        [
            "Single sequential baseline: fit gender → fit age → fit disease (no overlap); "
            "parallel design: three fits overlap → illustrates task parallelism.",
            "Random Forest n_jobs=-1: many trees built in parallel → illustrates embarrassingly "
            "parallel work inside one estimator.",
            "Inference: three predictions overlap → same idea at serving time.",
        ],
    )

    # ----- 4 ML Models -----
    heading(doc, "4. Machine Learning Models", 1)
    para(
        doc,
        "Gender and disease are modeled with RandomForestClassifier; age uses "
        "RandomForestRegressor. Gender labels are encoded with sklearn's LabelEncoder before "
        "training. The disease head outputs class probabilities used to rank conditions and to "
        "report percentage confidence.",
    )

    heading(doc, "4.1 Why Random Forest", 2)
    bullet_list(
        doc,
        [
            "Handles mixed nonlinear relationships between sensors and targets without manual "
            "feature crosses.",
            "Robust to noise and works well on tabular data; many trees reduce variance.",
            "Built-in parallelism via n_jobs aligns with the project's parallel computing goals.",
            "Predict_proba for disease gives interpretable probability masses over seven classes.",
        ],
    )

    # ----- 5 Implementation -----
    heading(doc, "5. Implementation Details", 1)
    para(
        doc,
        "The project is organized into dataset.py (data generation), model.py (training, "
        "evaluation, saving, parallel prediction, ASCII visualization on the console), and "
        "optional virtual environment venv_pdc for dependency isolation. Paths are resolved "
        "relative to the script directory so that CSV and .pkl files live alongside the code.",
    )
    para(
        doc,
        "Important dependencies include pandas, numpy, scikit-learn, and joblib. The training "
        "split uses train_test_split with test_size=0.2 and random_state=42 for reproducibility.",
    )

    heading(doc, "5.1 Parallel Training (Code Snippet)", 2)
    add_code_block(
        doc,
        """
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    f1 = executor.submit(model_gender.fit, X_train, yg_train)
    f2 = executor.submit(model_age.fit, X_train, ya_train)
    f3 = executor.submit(model_disease.fit, X_train, yd_train)
    f1.result()
    f2.result()
    f3.result()
""",
    )

    heading(doc, "5.2 predict_parallel Function (Code Snippet)", 2)
    add_code_block(
        doc,
        """
def predict_parallel(sensor_values, features, model_gender, model_age,
                     model_disease, le_gender):
    X_in = pd.DataFrame(sensor_values, columns=features)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        fut_g = executor.submit(model_gender.predict, X_in)
        fut_a = executor.submit(model_age.predict, X_in)
        fut_d = executor.submit(model_disease.predict_proba, X_in)
        gender = le_gender.inverse_transform(fut_g.result())[0]
        age = round(float(fut_a.result()[0]))
        probs = fut_d.result()[0]
        ...
    return { "Gender": gender, "Predicted Age": age, "Top Disease": top_disease, ... }
""",
    )

    heading(doc, "5.3 Project Files and Artifacts", 2)
    para(
        doc,
        "Table 4 lists the main files. The virtual environment venv_pdc is optional but "
        "recommended so that package versions match across machines.",
    )
    t4 = doc.add_table(rows=1, cols=2)
    t4.style = "Table Grid"
    t4.rows[0].cells[0].text = "File"
    t4.rows[0].cells[1].text = "Description"
    for fname, desc in [
        ("dataset.py", "Generates health_dataset_20000.csv with reproducible seed."),
        ("model.py", "Trains models, evaluates, saves .pkl files, runs parallel prediction."),
        ("build_pdc_report.py", "Builds PDC_Health_Diagnosis_Report.docx; embeds plot_outputs/*.png."),
        ("health_dataset_20000.csv", "Tabular dataset used for training and testing."),
        ("plot_outputs/*.png", "Bar chart, confusion matrices, optional seaborn heatmaps (after model.py)."),
        ("model_gender.pkl, model_age.pkl, model_disease.pkl", "Serialized trained estimators."),
        ("gender_encoder.pkl", "LabelEncoder for gender saved to decode predictions."),
    ]:
        row = t4.add_row().cells
        row[0].text = fname
        row[1].text = desc
    doc.add_paragraph()
    doc.add_paragraph("Table 4: Project artifacts.").runs[0].italic = True

    # ----- 6 Results -----
    heading(doc, "6. Results and Discussion", 1)
    para(
        doc,
        "On a typical run after regenerating the dataset, test-set performance is in the following "
        "range (exact decimals may vary slightly if the CSV is regenerated without the same "
        "parameters): gender classification accuracy about 99%; disease classification accuracy "
        "about 92%; age regression R-squared about 0.81 with MAE near 5.8 years. These figures "
        "appear in the console together with ASCII bar charts and confusion matrices for disease "
        "and gender.",
    )

    heading(doc, "6.1 Training Output and Parallel Messages", 2)
    add_code_block(
        doc,
        """
Training three models in parallel (ThreadPoolExecutor, max_workers=3).
Each model uses n_jobs=-1 inside RandomForest.

All models trained!

=== Test set scores (same split, random_state=42) ===
Gender classification accuracy: 99.15%
Disease classification accuracy: 91.65%
Age regression R-squared: 0.8109 (variance explained)
Age regression MAE (years): 5.80
""",
    )

    heading(doc, "6.2 Example Prediction (Typhoid-like Sensors)", 2)
    add_code_block(
        doc,
        """
Input row: [4.8, 1.1, 0.4, 4.3, 7.2]

Prediction result:
  Gender: Male
  Predicted Age: 23
  Top Disease: Typhoid
  All disease probabilities (%):
    - Typhoid: 62.0
    - Malaria: 36.0
    - Dengue: 2.0
    ...
""",
    )
    para(
        doc,
        "The probability vector shows how the forest distributes mass over competing diseases; "
        "the argmax label is reported as Top Disease. This supports decision support rather than "
        "replacing clinical diagnosis.",
    )

    heading(doc, "6.3 Confusion Matrices and Error Patterns", 2)
    para(
        doc,
        "The program prints confusion matrices for disease and gender on the test split. For "
        "disease, most mass lies on the diagonal, but some confusion appears between clinically "
        "similar conditions (for example Typhoid versus Malaria versus Dengue when fever and "
        "weakness are high). For gender, the matrix is nearly diagonal because the synthetic "
        "generator injects a consistent cough and sneeze pattern by gender. These tables help "
        "explain residual errors and guide future data collection.",
    )
    para(
        doc,
        "The same matrices appear as numeric tables and ASCII heatmaps in the terminal when you "
        "run model.py; the exported PNG files under plot_outputs/ (see Section 6.4) provide "
        "report-ready figures. For the viva, you can point to diagonal dominance versus off-diagonal "
        "confusion cells to explain per-class behavior.",
    )

    plot_dir = path.parent / "plot_outputs"
    heading(doc, "6.4 Figures: Metrics Bar Chart and Confusion Matrix Screenshots", 2)
    para(
        doc,
        "After running python model.py, matplotlib and (optionally) seaborn save images in the "
        "plot_outputs folder. Embed these in the printed report as evidence of evaluation: test "
        "scores (bar chart), raw-count confusion matrices, and—if seaborn is installed—row-normalized "
        "heatmaps and a feature-correlation heatmap. If any image is missing, run model.py again "
        "from the project directory with matplotlib installed; pip install seaborn adds three "
        "extra PNGs.",
    )
    fig_specs = [
        ("metrics_bar_chart.png", "Figure 1: Test set scores (gender and disease accuracy; age as R-squared × 100)."),
        ("disease_confusion_matrix.png", "Figure 2: Disease confusion matrix (counts, test set)—sklearn ConfusionMatrixDisplay."),
        ("gender_confusion_matrix.png", "Figure 3: Gender confusion matrix (counts, test set)."),
        (
            "disease_confusion_matrix_normalized.png",
            "Figure 4: Disease confusion matrix row-normalized (recall per true class)—seaborn.",
        ),
        (
            "gender_confusion_matrix_normalized.png",
            "Figure 5: Gender confusion matrix row-normalized—seaborn.",
        ),
        (
            "feature_correlation_heatmap.png",
            "Figure 6: Correlation heatmap among sensor and migraine features—seaborn.",
        ),
    ]
    any_fig = False
    for fname, cap in fig_specs:
        if add_figure(doc, plot_dir / fname, cap, width_inches=5.8 if "correlation" in fname else 6.2):
            any_fig = True
    if not any_fig:
        para(
            doc,
            "No PNG files were found in plot_outputs/. Generate them by running: python model.py "
            "(from the project folder). Then rebuild this report: python build_pdc_report.py",
            bold=False,
        )

    heading(doc, "6.5 Training Time Considerations", 2)
    para(
        doc,
        "Wall-clock training time depends on CPU core count, RAM bandwidth, and n_estimators. "
        "The report does not fix a single millisecond value because laboratory PCs differ; "
        "students should record their own timings with time.perf_counter() around the training "
        "block if required by the instructor. Qualitatively, parallel training should not exceed "
        "the sum of individual fits by more than the slowest model when cores are saturated, "
        "whereas sequential training always adds all three durations.",
    )

    # ----- 7 Challenges -----
    heading(doc, "7. Challenges Faced and Solutions", 1)
    bullet_list(
        doc,
        [
            "Balancing synthetic data so disease classes remain separable while age and gender "
            "remain learnable: addressed by tuning noise scales and small demographic adjustments.",
            "Windows console encoding (cp1252) and Unicode in plots: addressed by using ASCII-only "
            "charts in the terminal and plain English messages.",
            "Parallelism vs. Python GIL: Random Forest heavy lifting runs in native code; thread "
            "pool still helps overlap three models on many setups.",
            "Interpreting age as regression: reported R-squared and MAE rather than classification "
            "accuracy.",
        ],
    )

    # ----- 8 Conclusion -----
    heading(doc, "8. Conclusion and Future Scope", 1)
    para(
        doc,
        "The project demonstrates a multi-task health diagnosis pipeline with explicit parallel "
        "training and parallel inference, layered on top of internal parallelism in Random "
        "Forest. It connects classroom ideas in parallel and distributed computing to a concrete "
        "Python implementation.",
    )

    heading(doc, "8.1 Scaling with Spark or Dask", 2)
    para(
        doc,
        "For large hospitals or streaming telemetry, data could be ingested into Spark DataFrames "
        "or Dask arrays. Feature extraction could run in parallel partitions; model training "
        "could use distributed ML libraries or separate worker pools per task. Model serving could "
        "use Kubernetes replicas for each head (gender, age, disease) behind a single API "
        "gateway, which is distributed computing in the service sense.",
    )

    heading(doc, "8.2 Real Sensor Hardware", 2)
    para(
        doc,
        "Physical devices would provide time-series streams instead of single-row summaries. Edge "
        "devices might run lightweight models; a server could aggregate batches and run the full "
        "forest ensemble. Calibration, missing data, and patient safety regulations become "
        "primary engineering concerns beyond this coursework prototype.",
    )

    para(
        doc,
        "Future work includes collecting real labeled data, adding uncertainty quantification, "
        "deploying under HIPAA or local privacy rules, and benchmarking true end-to-end latency "
        "under concurrent client load.",
    )

    heading(doc, "Appendix A: Reproducibility Checklist", 1)
    bullet_list(
        doc,
        [
            "Use Python 3.10+ or 3.12 in a clean virtual environment.",
            "Install pandas, numpy, scikit-learn, and joblib.",
            "Run dataset.py before model.py so the CSV matches the generator logic.",
            "Keep random_state=42 in train_test_split and RandomForest for repeatable metrics.",
        ],
    )

    heading(doc, "Appendix B: Glossary", 1)
    bullet_list(
        doc,
        [
            "Task parallelism: independent tasks run concurrently (here, three model fits).",
            "Data parallelism: same operation on different data chunks (tree building in RF).",
            "Embarrassingly parallel: little or no dependency between parallel units.",
            "Latency: time to return one prediction; parallel inference reduces it when possible.",
        ],
    )

    # References placeholder
    heading(doc, "References", 1)
    bullet_list(
        doc,
        [
            "Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 2011.",
            "Breiman, L., Random Forests, Machine Learning, 2001.",
            "Python Software Foundation, concurrent.futures documentation.",
            "Zaharia et al., Apache Spark: A unified engine for big data processing, CACM 2016.",
            "Rocklin, M., Dask: Parallelism with analytics, 2015.",
        ],
    )

    doc.save(path)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "PDC_Health_Diagnosis_Report.docx"
    build_report(out)
    print("Saved:", out)
