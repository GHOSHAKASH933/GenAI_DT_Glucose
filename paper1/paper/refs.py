"""Reference database shared by both papers, with MDPI and Elsevier (numbered) formatters.
Each entry: au = list of "Initials Surname" strings, kind in {article, conf, book, arxiv, data, chapter}."""
from __future__ import annotations
import re

R = {
 "Liu2019": dict(kind="article", au=["Y. Liu", "L. Zhang", "Y. Yang", "L. Zhou", "L. Ren", "F. Wang", "R. Liu", "Z. Pang", "M.J. Deen"],
     title="A novel cloud-based framework for the elderly healthcare services using digital twin", j="IEEE Access", y=2019, v="7", p="49088--49101", doi="10.1109/ACCESS.2019.2909828"),
 "Barricelli2019": dict(kind="article", au=["B.R. Barricelli", "E. Casiraghi", "D. Fogli"],
     title="A survey on digital twin: Definitions, characteristics, applications, and design implications", j="IEEE Access", y=2019, v="7", p="167653--167671", doi="10.1109/ACCESS.2019.2953499"),
 "CorralAcero2020": dict(kind="article", au=["J. Corral-Acero", "F. Margara", "M. Marciniak", "C. Rodero", "F. Loncaric", "Y. Feng", "P. Lamata"],
     title="The `Digital Twin' to enable the vision of precision cardiology", j="Eur. Heart J.", y=2020, v="41", n="48", p="4556--4564", doi="10.1093/eurheartj/ehaa159"),
 "Karakra2018": dict(kind="conf", au=["A. Karakra", "F. Fontanili", "E. Lamine", "J. Lamothe", "A. Taweel"],
     title="Pervasive computing integrated discrete event simulation for a hospital digital twin", bt="Proceedings of the IEEE/ACS 15th International Conference on Computer Systems and Applications (AICCSA)", y=2018, p="1--6"),
 "Karakra2020": dict(kind="conf", au=["A. Karakra", "E. Lamine", "F. Fontanili", "J. Lamothe"],
     title="HospiT'Win: A digital twin framework for patients' pathways real-time monitoring and hospital organizational resilience capacity enhancement", bt="Proceedings of the International Workshop on Innovative Simulation for Healthcare", y=2020, p="62--71"),
 "Kovatchev2009": dict(kind="article", au=["B.P. Kovatchev", "M. Breton", "C. Dalla Man", "C. Cobelli"],
     title="In silico preclinical trials: A proof of concept in closed-loop control of type 1 diabetes", j="J. Diabetes Sci. Technol.", y=2009, v="3", n="1", p="44--55", doi="10.1177/193229680900300106"),
 "DallaMan2007": dict(kind="article", au=["C. Dalla Man", "R.A. Rizza", "C. Cobelli"],
     title="Meal simulation model of the glucose-insulin system", j="IEEE Trans. Biomed. Eng.", y=2007, v="54", n="10", p="1740--1749", doi="10.1109/TBME.2007.893506"),
 "DallaMan2014": dict(kind="article", au=["C. Dalla Man", "F. Micheletto", "D. Lv", "M. Breton", "B. Kovatchev", "C. Cobelli"],
     title="The UVA/PADOVA type 1 diabetes simulator: New features", j="J. Diabetes Sci. Technol.", y=2014, v="8", n="1", p="26--34", doi="10.1177/1932296813514502"),
 "simglucose": dict(kind="data", au=["J. Xie"], title="Simglucose v0.2.1: A Type-1 Diabetes simulator implemented in Python", pub="GitHub repository", y=2018, url="https://github.com/jxx123/simglucose"),
 "Hovorka2004": dict(kind="article", au=["R. Hovorka", "V. Canonico", "L.J. Chassin", "U. Haueter", "M. Massi-Benedetti", "M. Orsini Federici", "T.R. Pieber", "H.C. Schaller", "L. Schaupp", "T. Vering", "M.E. Wilinska"],
     title="Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes", j="Physiol. Meas.", y=2004, v="25", n="4", p="905--920", doi="10.1088/0967-3334/25/4/010"),
 "Battelino2019": dict(kind="article", au=["T. Battelino", "T. Danne", "R.M. Bergenstal", "S.A. Amiel", "R. Beck", "T. Biester"], etal=True,
     title="Clinical targets for continuous glucose monitoring data interpretation: Recommendations from the international consensus on time in range", j="Diabetes Care", y=2019, v="42", n="8", p="1593--1603", doi="10.2337/dci19-0028"),
 "Viceconti2020": dict(kind="article", au=["M. Viceconti", "M. De Vos", "S. Mellone", "L. Geris"],
     title="Position paper: Credibility of in silico trial technologies---A theoretical framing", j="IEEE J. Biomed. Health Inform.", y=2020, v="24", n="1", p="4--13", doi="10.1109/JBHI.2019.2949888"),
 "Bruynseels2018": dict(kind="article", au=["K. Bruynseels", "F. Santoni de Sio", "J. van den Hoven"],
     title="Digital twins in health care: Ethical implications of an emerging engineering paradigm", j="Front. Genet.", y=2018, v="9", p="31", doi="10.3389/fgene.2018.00031"),
 "Sohn2015": dict(kind="conf", au=["K. Sohn", "H. Lee", "X. Yan"], title="Learning structured output representation using deep conditional generative models",
     bt="Advances in Neural Information Processing Systems 28 (NeurIPS)", y=2015, p="3483--3491"),
 "Kingma2014": dict(kind="conf", au=["D.P. Kingma", "M. Welling"], title="Auto-encoding variational Bayes", bt="Proceedings of the 2nd International Conference on Learning Representations (ICLR)", y=2014),
 "Bowman2016": dict(kind="conf", au=["S.R. Bowman", "L. Vilnis", "O. Vinyals", "A.M. Dai", "R. Jozefowicz", "S. Bengio"], title="Generating sentences from a continuous space",
     bt="Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning (CoNLL)", y=2016, p="10--21"),
 "Alemi2018": dict(kind="conf", au=["A. Alemi", "B. Poole", "I. Fischer", "J. Dillon", "R.A. Saurous", "K. Murphy"], title="Fixing a broken ELBO",
     bt="Proceedings of the 35th International Conference on Machine Learning (ICML)", y=2018, p="159--168"),
 "Lucas2019": dict(kind="conf", au=["J. Lucas", "G. Tucker", "R. Grosse", "M. Norouzi"], title="Don't blame the ELBO! A linear VAE perspective on posterior collapse",
     bt="Advances in Neural Information Processing Systems 32 (NeurIPS)", y=2019),
 "Vovk2005": dict(kind="book", au=["V. Vovk", "A. Gammerman", "G. Shafer"], title="Algorithmic Learning in a Random World", pub="Springer", addr="New York, NY, USA", y=2005),
 "Angelopoulos2021": dict(kind="arxiv", au=["A.N. Angelopoulos", "S. Bates"], title="A gentle introduction to conformal prediction and distribution-free uncertainty quantification", arx="2107.07511", y=2021),
 "McMahan2017": dict(kind="conf", au=["H.B. McMahan", "E. Moore", "D. Ramage", "S. Hampson", "B. Ag\\\"uera y Arcas"], title="Communication-efficient learning of deep networks from decentralized data",
     bt="Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)", y=2017, p="1273--1282"),
 "Hinton2015": dict(kind="arxiv", au=["G. Hinton", "O. Vinyals", "J. Dean"], title="Distilling the knowledge in a neural network", arx="1503.02531", y=2015),
 "Wilcoxon1945": dict(kind="article", au=["F. Wilcoxon"], title="Individual comparisons by ranking methods", j="Biometrics Bull.", y=1945, v="1", n="6", p="80--83", doi="10.2307/3001968"),
 "Holm1979": dict(kind="article", au=["S. Holm"], title="A simple sequentially rejective multiple test procedure", j="Scand. J. Stat.", y=1979, v="6", n="2", p="65--70"),
 "Pearl2009": dict(kind="book", au=["J. Pearl"], title="Causality: Models, Reasoning, and Inference, 2nd ed.", pub="Cambridge University Press", addr="Cambridge, UK", y=2009),
 "Hernan2020": dict(kind="book", au=["M.A. Hern\\'an", "J.M. Robins"], title="Causal Inference: What If", pub="Chapman \\& Hall/CRC", addr="Boca Raton, FL, USA", y=2020),
 "Shalit2017": dict(kind="conf", au=["U. Shalit", "F.D. Johansson", "D. Sontag"], title="Estimating individual treatment effect: Generalization bounds and algorithms",
     bt="Proceedings of the 34th International Conference on Machine Learning (ICML)", y=2017, p="3076--3085"),
 "Lim2018": dict(kind="conf", au=["B. Lim", "A. Alaa", "M. van der Schaar"], title="Forecasting treatment responses over time using recurrent marginal structural networks",
     bt="Advances in Neural Information Processing Systems 31 (NeurIPS)", y=2018),
 "Bica2020": dict(kind="conf", au=["I. Bica", "A.M. Alaa", "J. Jordon", "M. van der Schaar"], title="Estimating counterfactual treatment outcomes over time through adversarially balanced representations",
     bt="Proceedings of the 8th International Conference on Learning Representations (ICLR)", y=2020),
 "Lei2018": dict(kind="article", au=["J. Lei", "M. G'Sell", "A. Rinaldo", "R.J. Tibshirani", "L. Wasserman"], title="Distribution-free predictive inference for regression",
     j="J. Am. Stat. Assoc.", y=2018, v="113", n="523", p="1094--1111", doi="10.1080/01621459.2017.1307116"),
 "Tibshirani2019": dict(kind="conf", au=["R.J. Tibshirani", "R. Foygel Barber", "E.J. Cand\\`es", "A. Ramdas"], title="Conformal prediction under covariate shift",
     bt="Advances in Neural Information Processing Systems 32 (NeurIPS)", y=2019),
 "Romano2019": dict(kind="conf", au=["Y. Romano", "E. Patterson", "E.J. Cand\\`es"], title="Conformalized quantile regression",
     bt="Advances in Neural Information Processing Systems 32 (NeurIPS)", y=2019),
 "Chow1970": dict(kind="article", au=["C.K. Chow"], title="On optimum recognition error and reject tradeoff", j="IEEE Trans. Inf. Theory", y=1970, v="16", n="1", p="41--46", doi="10.1109/TIT.1970.1054406"),
 "Geifman2017": dict(kind="conf", au=["Y. Geifman", "R. El-Yaniv"], title="Selective classification for deep neural networks", bt="Advances in Neural Information Processing Systems 30 (NeurIPS)", y=2017),
 "Doyle2014": dict(kind="article", au=["F.J. Doyle III", "L.M. Huyett", "J.B. Lee", "H.C. Zisser", "E. Dassau"], title="Closed-loop artificial pancreas systems: Engineering the algorithms",
     j="Diabetes Care", y=2014, v="37", n="5", p="1191--1197", doi="10.2337/dc13-2108"),
 "Bergman1979": dict(kind="article", au=["R.N. Bergman", "Y.Z. Ider", "C.R. Bowden", "C. Cobelli"], title="Quantitative estimation of insulin sensitivity",
     j="Am. J. Physiol.", y=1979, v="236", n="6", p="E667--E677", doi="10.1152/ajpendo.1979.236.6.E667"),
 "Venkatesh2022": dict(kind="article", au=["K.P. Venkatesh", "M.M. Raza", "J.C. Kvedar"], title="Health digital twins as tools for precision medicine: Considerations for computation, implementation, and regulation",
     j="NPJ Digit. Med.", y=2022, v="5", p="150", doi="10.1038/s41746-022-00694-7"),
 "Coorey2022": dict(kind="article", au=["G. Coorey", "G.A. Figtree", "D.F. Fletcher", "V.J. Snelson", "S.T. Vernon", "D. Winlaw"], etal=True,
     title="The health digital twin to tackle cardiovascular disease---A review of an emerging interdisciplinary field", j="NPJ Digit. Med.", y=2022, v="5", p="126", doi="10.1038/s41746-022-00640-7"),
 "Katsoulakis2024": dict(kind="article", au=["E. Katsoulakis", "Q. Wang", "H. Wu", "L. Shahriyari", "R. Fletcher", "J. Liu"], etal=True,
     title="Digital twins for health: A scoping review", j="NPJ Digit. Med.", y=2024, v="7", p="77", doi="10.1038/s41746-024-01073-0"),
 "Bordukova2023": dict(kind="article", au=["M. Bordukova", "N. Makarov", "R. Rodriguez-Esteban", "F. Schmich", "M.P. Menden"],
     title="Generative artificial intelligence empowers digital twins in drug discovery and clinical trials", j="Expert Opin. Drug Discov.", y=2024, v="19", n="1", p="33--42", doi="10.1080/17460441.2023.2273839"),
 "Vallee2023": dict(kind="article", au=["A. Vall\\'ee"], title="Digital twin for healthcare systems", j="Front. Digit. Health", y=2023, v="5", p="1253050", doi="10.3389/fdgth.2023.1253050"),
 "Sun2022": dict(kind="article", au=["T. Sun", "X. He", "X. Song", "L. Shu", "Z. Li"], title="The digital twin in medicine: A key to the future of healthcare?",
     j="Front. Med.", y=2022, v="9", p="907066", doi="10.3389/fmed.2022.907066"),
 "Erol2020": dict(kind="conf", au=["T. Erol", "A.F. Mendi", "D. Do\\u{g}an"], title="The digital twin revolution in healthcare",
     bt="Proceedings of the 4th International Symposium on Multidisciplinary Studies and Innovative Technologies (ISMSIT)", y=2020, p="1--7"),
 "Sahal2022": dict(kind="article", au=["R. Sahal", "S.H. Alsamhi", "K.N. Brown"], title="Personal digital twin: A close look into the present and a step towards the future of personalised healthcare industry",
     j="Sensors", y=2022, v="22", n="15", p="5918", doi="10.3390/s22155918"),
 "Okegbile2022": dict(kind="conf", au=["S.D. Okegbile", "J. Cai"], title="Edge-assisted human-to-virtual twin connectivity scheme for human digital twin frameworks",
     bt="Proceedings of the IEEE 95th Vehicular Technology Conference (VTC2022-Spring)", y=2022, p="1--6"),
 "Ahmed2021": dict(kind="article", au=["H. Ahmed", "L. Devoto"], title="The potential of a digital twin in surgery", j="Surg. Innov.", y=2021, v="28", n="4", p="509--510", doi="10.1177/1553350620975896"),
 "Li2020": dict(kind="conf", au=["T. Li", "A.K. Sahu", "M. Zaheer", "M. Sanjabi", "A. Talwalkar", "V. Smith"], title="Federated optimization in heterogeneous networks",
     bt="Proceedings of Machine Learning and Systems 2 (MLSys)", y=2020, p="429--450"),
 "Fallah2020": dict(kind="conf", au=["A. Fallah", "A. Mokhtari", "A. Ozdaglar"], title="Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach",
     bt="Advances in Neural Information Processing Systems 33 (NeurIPS)", y=2020),
 "Rieke2020": dict(kind="article", au=["N. Rieke", "J. Hancox", "W. Li", "F. Milletar\\`i", "H.R. Roth", "S. Albarqouni"], etal=True,
     title="The future of digital health with federated learning", j="NPJ Digit. Med.", y=2020, v="3", p="119", doi="10.1038/s41746-020-00323-1"),
 "Zhao2023": dict(kind="article", au=["Q. Zhao", "J. Zhu", "X. Shen", "C. Lin", "Y. Zhang", "Y. Liang", "B. Cao", "J. Li", "X. Liu", "W. Rao", "C. Wang"],
     title="Chinese diabetes datasets for data-driven machine learning", j="Sci. Data", y=2023, v="10", p="35", doi="10.1038/s41597-023-01940-7"),
 "Efron1993": dict(kind="book", au=["B. Efron", "R.J. Tibshirani"], title="An Introduction to the Bootstrap", pub="Chapman \\& Hall", addr="New York, NY, USA", y=1993),
 "Clarke1987": dict(kind="article", au=["W.L. Clarke", "D. Cox", "L.A. Gonder-Frederick", "W. Carter", "S.L. Pohl"], title="Evaluating clinical accuracy of systems for self-monitoring of blood glucose",
     j="Diabetes Care", y=1987, v="10", n="5", p="622--628", doi="10.2337/diacare.10.5.622"),
 "Gneiting2007": dict(kind="article", au=["T. Gneiting", "A.E. Raftery"], title="Strictly proper scoring rules, prediction, and estimation",
     j="J. Am. Stat. Assoc.", y=2007, v="102", n="477", p="359--378", doi="10.1198/016214506000001437"),
 "Lakshminarayanan2017": dict(kind="conf", au=["B. Lakshminarayanan", "A. Pritzel", "C. Blundell"], title="Simple and scalable predictive uncertainty estimation using deep ensembles",
     bt="Advances in Neural Information Processing Systems 30 (NeurIPS)", y=2017),
 "Jacob2018": dict(kind="conf", au=["B. Jacob", "S. Kligys", "B. Chen", "M. Zhu", "M. Tang", "A. Howard", "H. Adam", "D. Kalenichenko"],
     title="Quantization and training of neural networks for efficient integer-arithmetic-only inference", bt="Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)", y=2018, p="2704--2713"),
 "Rosenbaum1983": dict(kind="article", au=["P.R. Rosenbaum", "D.B. Rubin"], title="The central role of the propensity score in observational studies for causal effects",
     j="Biometrika", y=1983, v="70", n="1", p="41--55", doi="10.1093/biomet/70.1.41"),
 "Cole2008": dict(kind="article", au=["S.R. Cole", "M.A. Hern\\'an"], title="Constructing inverse probability weights for marginal structural models",
     j="Am. J. Epidemiol.", y=2008, v="168", n="6", p="656--664", doi="10.1093/aje/kwn164"),
 "Petersen2012": dict(kind="article", au=["M.L. Petersen", "K.E. Porter", "S. Gruber", "Y. Wang", "M.J. van der Laan"], title="Diagnosing and responding to violations in the positivity assumption",
     j="Stat. Methods Med. Res.", y=2012, v="21", n="1", p="31--54", doi="10.1177/0962280210386207"),
 "DAmour2021": dict(kind="article", au=["A. D'Amour", "P. Ding", "A. Feller", "L. Lei", "J. Sekhon"], title="Overlap in observational studies with high-dimensional covariates",
     j="J. Econom.", y=2021, v="221", n="2", p="644--654", doi="10.1016/j.jeconom.2019.10.014"),
 "Schulam2017": dict(kind="conf", au=["P. Schulam", "S. Saria"], title="Reliable decision support using counterfactual models",
     bt="Advances in Neural Information Processing Systems 30 (NeurIPS)", y=2017),
 "Li2020glu": dict(kind="article", au=["K. Li", "J. Daniels", "C. Liu", "P. Herrero", "P. Georgiou"], title="Convolutional recurrent neural networks for glucose prediction",
     j="IEEE J. Biomed. Health Inform.", y=2020, v="24", n="2", p="603--613", doi="10.1109/JBHI.2019.2908488"),
 "Marling2020": dict(kind="conf", au=["C. Marling", "R. Bunescu"], title="The OhioT1DM dataset for blood glucose level prediction: Update 2020",
     bt="Proceedings of the 5th International Workshop on Knowledge Discovery in Healthcare Data (KDH), CEUR Workshop Proceedings 2675", y=2020, p="71--74"),
 "Cho2014": dict(kind="conf", au=["K. Cho", "B. van Merri\\\"enboer", "C. Gulcehre", "D. Bahdanau", "F. Bougares", "H. Schwenk", "Y. Bengio"],
     title="Learning phrase representations using RNN encoder--decoder for statistical machine translation", bt="Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)", y=2014, p="1724--1734"),
 "Xie2020": dict(kind="article", au=["J. Xie", "Q. Wang"], title="Benchmarking machine learning algorithms on blood glucose prediction for type I diabetes in comparison with classical time-series models",
     j="IEEE Trans. Biomed. Eng.", y=2020, v="67", n="11", p="3101--3124", doi="10.1109/TBME.2020.2975959"),
 "Oviedo2017": dict(kind="article", au=["S. Oviedo", "J. Veh\\'i", "R. Calm", "J. Armengol"], title="A review of personalized blood glucose prediction strategies for T1DM patients",
     j="Int. J. Numer. Methods Biomed. Eng.", y=2017, v="33", n="6", p="e2833", doi="10.1002/cnm.2833"),
 "Nix1994": dict(kind="conf", au=["D.A. Nix", "A.S. Weigend"], title="Estimating the mean and variance of the target probability distribution",
     bt="Proceedings of the IEEE International Conference on Neural Networks (ICNN)", y=1994, p="55--60"),
 "Szekely2013": dict(kind="article", au=["G.J. Sz\\'ekely", "M.L. Rizzo"], title="Energy statistics: A class of statistics based on distances",
     j="J. Stat. Plan. Inference", y=2013, v="143", n="8", p="1249--1272", doi="10.1016/j.jspi.2013.03.018"),
 "Kingma2016": dict(kind="conf", au=["D.P. Kingma", "T. Salimans", "R. Jozefowicz", "X. Chen", "I. Sutskever", "M. Welling"], title="Improved variational inference with inverse autoregressive flow",
     bt="Advances in Neural Information Processing Systems 29 (NeurIPS)", y=2016),
 "Vidovszky2024": dict(kind="article", au=["A.A. Vidovszky", "C.K. Fisher", "A.D. Loukianov", "A.M. Smith", "E.W. Tramel", "J.R. Walsh", "J.L. Ross"],
     title="Increasing acceptance of AI-generated digital twins through clinical trial applications", j="Clin. Transl. Sci.", y=2024, v="17", n="7", p="e13897", doi="10.1111/cts.13897"),
 "Iqbal2022": dict(kind="article", au=["J.D. Iqbal", "M. Krauthammer", "N. Biller-Andorno"], title="The use and ethics of digital twins in medicine",
     j="J. Law Med. Ethics", y=2022, v="50", n="3", p="583--596", doi="10.1017/jme.2022.97"),
 "Zhao2022data": dict(kind="data", au=["Q. Zhao", "J. Zhu", "X. Shen", "C. Lin", "Y. Zhang", "Y. Liang", "B. Cao", "J. Li", "X. Liu", "W. Rao", "C. Wang"],
     title="Diabetes datasets: ShanghaiT1DM and ShanghaiT2DM", pub="figshare", y=2023, url="https://doi.org/10.6084/m9.figshare.c.6310860"),
 "Chen2024": dict(kind="arxiv", au=["J. Chen", "Y. Shi", "C. Yi", "H. Du", "J. Kang", "D. Niyato"], title="Generative AI-driven human digital twin in IoT-healthcare: A comprehensive survey", arx="2401.13699", y=2024),
}


def _mdpi_name(a):
    parts = a.split(" ")
    ini, sur = parts[0], " ".join(parts[1:])
    return f"{sur}, {ini}"


def mdpi(k):
    e = R[k]
    au = "; ".join(_mdpi_name(a) for a in e["au"])
    if e.get("etal"):
        au += "; et al"
    t = e["title"]
    if e["kind"] == "article":
        s = f"{au}. {t}. \\emph{{{e['j']}}} \\textbf{{{e['y']}}}, \\emph{{{e['v']}}}, {e['p']}."
        if e.get("doi"):
            s += f" \\url{{https://doi.org/{e['doi']}}}."
    elif e["kind"] == "conf":
        s = f"{au}. {t}. In {e['bt']}, {e['y']}" + (f"; pp. {e['p']}." if e.get("p") else ".")
    elif e["kind"] == "book":
        s = f"{au}. \\emph{{{t}}}; {e['pub']}: {e['addr']}, {e['y']}."
    elif e["kind"] == "arxiv":
        s = f"{au}. {t}. \\emph{{arXiv}} \\textbf{{{e['y']}}}, arXiv:{e['arx']}."
    else:
        s = f"{au}. {t}. {e['pub']}, {e['y']}. Available online: \\url{{{e['url']}}}."
    return s.replace("..", ".")


def els(k):
    e = R[k]
    au = e["au"]
    names = ", ".join(au) + (", et al." if e.get("etal") else "")
    t = e["title"]
    if e["kind"] == "article":
        iss = f" ({e['n']})" if e.get("n") else ""
        s = f"{names}, {t}, {e['j']} {e['v']} ({e['y']}) {e['p']}."
        if e.get("doi"):
            s += f" \\url{{https://doi.org/{e['doi']}}}."
    elif e["kind"] == "conf":
        s = f"{names}, {t}, in: {e['bt']}, {e['y']}" + (f", pp. {e['p']}." if e.get("p") else ".")
    elif e["kind"] == "book":
        s = f"{names}, {t}, {e['pub']}, {e['addr']}, {e['y']}."
    elif e["kind"] == "arxiv":
        s = f"{names}, {t}, arXiv preprint arXiv:{e['arx']} ({e['y']})."
    else:
        s = f"{names}, {t}, {e['pub']} ({e['y']}). \\url{{{e['url']}}}."
    return s.replace("..", ".")


def order(tex):
    """Keys in order of first citation."""
    seen = []
    for grp in re.findall(r"\\cite[pt]?\{([^}]*)\}", tex):
        for k in grp.split(","):
            k = k.strip()
            if k not in seen:
                seen.append(k)
    return seen


def bibliography(tex, style="mdpi"):
    keys = order(tex)
    missing = [k for k in keys if k not in R]
    if missing:
        raise KeyError(f"missing references: {missing}")
    f = mdpi if style == "mdpi" else els
    w = "999" if style == "mdpi" else "00"
    items = "\n".join(f"\\bibitem{{{k}}}\n{f(k)}\n" for k in keys)
    return f"\\begin{{thebibliography}}{{{w}}}\n{items}\\end{{thebibliography}}"


# ---- added for the extended Paper 1 benchmark (transformer, diffusion, quantile and calibration references)
R.update({
 "Vaswani2017": dict(kind="conf", au=["A. Vaswani", "N. Shazeer", "N. Parmar", "J. Uszkoreit", "L. Jones", "A.N. Gomez", "\\L. Kaiser", "I. Polosukhin"],
     title="Attention is all you need", bt="Advances in Neural Information Processing Systems 30 (NeurIPS 2017)", y=2017, p="5998--6008"),
 "Ho2020": dict(kind="conf", au=["J. Ho", "A. Jain", "P. Abbeel"],
     title="Denoising diffusion probabilistic models", bt="Advances in Neural Information Processing Systems 33 (NeurIPS 2020)", y=2020, p="6840--6851"),
 "Song2021": dict(kind="arxiv", au=["J. Song", "C. Meng", "S. Ermon"],
     title="Denoising diffusion implicit models", y=2020, arx="2010.02502"),
 "Romano2019": dict(kind="conf", au=["Y. Romano", "E. Patterson", "E.J. Cand\\`es"],
     title="Conformalized quantile regression", bt="Advances in Neural Information Processing Systems 32 (NeurIPS 2019)", y=2019, p="3543--3553"),
 "Tashiro2021": dict(kind="conf", au=["Y. Tashiro", "J. Song", "Y. Song", "S. Ermon"],
     title="CSDI: Conditional score-based diffusion models for probabilistic time series imputation", bt="Advances in Neural Information Processing Systems 34 (NeurIPS 2021)", y=2021, p="24804--24816"),
 "Rasul2021": dict(kind="conf", au=["K. Rasul", "C. Seward", "I. Schuster", "R. Vollgraf"],
     title="Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting", bt="Proceedings of the 38th International Conference on Machine Learning (PMLR 139)", y=2021, p="8857--8868"),
 "Nie2023": dict(kind="arxiv", au=["Y. Nie", "N.H. Nguyen", "P. Sinthong", "J. Kalagnanam"],
     title="A time series is worth 64 words: Long-term forecasting with transformers", y=2022, arx="2211.14730"),
 "Zeng2023": dict(kind="article", au=["A. Zeng", "M. Chen", "L. Zhang", "Q. Xu"],
     title="Are transformers effective for time series forecasting?", j="Proc. AAAI Conf. Artif. Intell.", y=2023, v="37", n="9", p="11121--11128", doi="10.1609/aaai.v37i9.26317"),
 "Lim2021": dict(kind="article", au=["B. Lim", "S.\\\"O. Ar{\\i}k", "N. Loeff", "T. Pfister"],
     title="Temporal fusion transformers for interpretable multi-horizon time series forecasting", j="Int. J. Forecast.", y=2021, v="37", n="4", p="1748--1764", doi="10.1016/j.ijforecast.2021.03.012"),
 "Koenker1978": dict(kind="article", au=["R. Koenker", "G. Bassett"],
     title="Regression quantiles", j="Econometrica", y=1978, v="46", n="1", p="33--50", doi="10.2307/1913643"),
 "Brier1950": dict(kind="article", au=["G.W. Brier"],
     title="Verification of forecasts expressed in terms of probability", j="Mon. Weather Rev.", y=1950, v="78", n="1", p="1--3"),
 "Naeini2015": dict(kind="conf", au=["M.P. Naeini", "G.F. Cooper", "M. Hauskrecht"],
     title="Obtaining well calibrated probabilities using Bayesian binning", bt="Proceedings of the 29th AAAI Conference on Artificial Intelligence", y=2015, p="2901--2907"),
 "Ansari2024": dict(kind="arxiv", au=["A.F. Ansari", "L. Stella", "C. Turkmen", "X. Zhang", "P. Mercado", "H. Shen"], etal=True,
     title="Chronos: Learning the language of time series", y=2024, arx="2403.07815"),
 "Karagoz2025": dict(kind="arxiv", au=["M.A. Karagoz", "M.D. Breton", "A. El Fathi"],
     title="A comparative study of transformer-based models for multi-horizon blood glucose prediction", y=2025, arx="2505.08821"),
 "Sun2024bayes": dict(kind="arxiv", au=["Y. Sun", "P. Kosmas"],
     title="Integrating Bayesian approaches and expert knowledge for forecasting continuous glucose monitoring values in type 2 diabetes mellitus", y=2024, arx="2409.07315"),
 "Prioleau2025": dict(kind="arxiv", au=["T. Prioleau", "B. Lu", "Y. Cui"],
     title="Glucose-ML: A collection of longitudinal diabetes datasets for development of robust AI solutions", y=2025, arx="2507.14077"),
 "Lutsker2024": dict(kind="arxiv", au=["G. Lutsker", "G. Sapir", "S. Shilo", "J. Merino", "A. Godneva", "J.R. Greenfield"], etal=True,
     title="From glucose patterns to health outcomes: A generalizable foundation model for continuous glucose monitor data analysis", y=2024, arx="2408.11876"),
})
