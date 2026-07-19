from dataclasses import dataclass


@dataclass(frozen=True)
class LabelSpec:
    id: str
    source_column: str
    title: str
    zh: str
    terms: tuple[str, ...]
    definition: str
    imaging: str
    anatomy_regions: tuple[str, ...]
    positive_threshold: float = 0.5
    uncertain_threshold: float = 0.35


LABEL_SPECS: tuple[LabelSpec, ...] = (
    LabelSpec(
        "medical_material",
        "Medical material",
        "Medical material",
        "医疗器械或植入物",
        ("medical material", "catheter", "pacemaker", "port", "prosthesis", "implant"),
        "Medical material denotes an implanted or externally introduced device visible on CT.",
        "CT can show catheters, ports, cardiac devices, surgical clips, and prostheses.",
        ("mediastinum", "heart", "chest_wall"),
    ),
    LabelSpec(
        "arterial_wall_calcification",
        "Arterial wall calcification",
        "Arterial wall calcification",
        "动脉壁钙化",
        (
            "arterial wall calcification",
            "aortic calcification",
            "calcific plaque",
            "aortic plaque",
            "calcified atherosclerotic change",
            "calcified atherosclerotic changes",
            "aortic atheroma plaque",
            "aortic atheroma plaques",
            "atheromatous plaque in the aorta",
            "atheromatous plaques in the aorta",
        ),
        "Arterial wall calcification is mineralization of an arterial wall or plaque.",
        "CT shows high-attenuation foci along the thoracic aorta or other arterial walls.",
        ("aorta", "mediastinum"),
    ),
    LabelSpec(
        "cardiomegaly",
        "Cardiomegaly",
        "Cardiomegaly",
        "心脏增大",
        ("cardiomegaly", "enlarged heart", "cardiac enlargement", "heart is enlarged"),
        "Cardiomegaly denotes enlargement of the heart.",
        "CT assessment uses chamber size and the relationship between the heart and thoracic cavity.",
        ("heart", "mediastinum"),
    ),
    LabelSpec(
        "pericardial_effusion",
        "Pericardial effusion",
        "Pericardial effusion",
        "心包积液",
        ("pericardial effusion", "pericardial fluid"),
        "Pericardial effusion is abnormal fluid in the pericardial space.",
        "CT shows fluid attenuation surrounding part or all of the heart.",
        ("pericardium", "heart"),
    ),
    LabelSpec(
        "coronary_artery_wall_calcification",
        "Coronary artery wall calcification",
        "Coronary artery wall calcification",
        "冠状动脉壁钙化",
        (
            "coronary artery calcification",
            "coronary calcification",
            "calcified coronary plaque",
            "atheroma plaques in the aorta and coronary arteries",
            "atheromatous plaques in the aorta and coronary arteries",
            "atherosclerotic changes in the coronary arteries",
        ),
        "Coronary artery wall calcification is calcified atherosclerotic plaque in a coronary artery.",
        "CT shows high-attenuation foci following the course of coronary arteries.",
        ("coronary_arteries", "heart"),
    ),
    LabelSpec(
        "hiatal_hernia",
        "Hiatal hernia",
        "Hiatal hernia",
        "食管裂孔疝",
        ("hiatal hernia", "hiatus hernia", "intrathoracic stomach"),
        "Hiatal hernia is herniation of stomach through the esophageal hiatus.",
        "CT can show stomach or gastroesophageal junction above the diaphragm.",
        ("esophagus", "diaphragm", "lower_mediastinum"),
    ),
    LabelSpec(
        "lymphadenopathy",
        "Lymphadenopathy",
        "Lymphadenopathy",
        "淋巴结肿大",
        (
            "lymphadenopathy",
            "enlarged lymph node",
            "enlarged lymph nodes",
            "adenopathy",
            "lymph node",
            "lymph nodes",
        ),
        "Lymphadenopathy denotes abnormal lymph-node enlargement.",
        "CT evaluates mediastinal, hilar, axillary, and supraclavicular nodal stations.",
        ("mediastinal_lymph_nodes", "hilar_lymph_nodes", "axillary_lymph_nodes"),
    ),
    LabelSpec(
        "emphysema",
        "Emphysema",
        "Emphysema",
        "肺气肿",
        ("emphysema", "emphysematous", "bullous change", "bullae"),
        "Emphysema is permanent enlargement of air spaces with destruction of alveolar walls.",
        "CT shows low-attenuation areas, vascular attenuation, and possible bullae.",
        ("right_lung", "left_lung", "lung_parenchyma"),
    ),
    LabelSpec(
        "atelectasis",
        "Atelectasis",
        "Atelectasis",
        "肺不张",
        ("atelectasis", "atelectatic", "collapse", "volume loss", "linear opacity"),
        "Atelectasis is partial or complete collapse of lung tissue.",
        "CT signs include volume loss, linear opacity, and displacement of fissures or vessels.",
        ("right_lung", "left_lung", "lung_lobes"),
    ),
    LabelSpec(
        "pulmonary_nodule",
        "Lung nodule",
        "Pulmonary nodule",
        "肺结节",
        ("pulmonary nodule", "lung nodule", "nodule", "nodular opacity", "nodules"),
        "A pulmonary nodule is a focal rounded opacity in the lung parenchyma.",
        "CT characterizes location, size, attenuation, margin, and growth of a focal opacity.",
        ("right_lung", "left_lung", "lung_lobes"),
    ),
    LabelSpec(
        "lung_opacity",
        "Lung opacity",
        "Lung opacity",
        "肺部密度增高影",
        (
            "lung opacity",
            "pulmonary opacity",
            "parenchymal opacity",
            "infiltrative lesion",
            "infiltrate",
            "ground-glass appearance",
            "ground-glass appearances",
            "ground glass area",
            "ground glass areas",
            "frosted glass area",
            "frosted glass areas",
            "pleuroparenchymal opacity",
            "pleuroparenchymal opacities",
        ),
        "Lung opacity is a nonspecific increase in pulmonary attenuation.",
        "CT appearance ranges from ground-glass attenuation to dense air-space opacity.",
        ("right_lung", "left_lung", "lung_parenchyma"),
    ),
    LabelSpec(
        "pulmonary_fibrotic_sequela",
        "Pulmonary fibrotic sequela",
        "Pulmonary fibrotic sequela",
        "肺纤维化后遗改变",
        (
            "pulmonary fibrosis",
            "fibrotic sequela",
            "fibrotic change",
            "fibrosis",
            "scarring",
            "fibrotic band",
            "fibrotic bands",
            "pleuroparenchymal band",
            "pleuroparenchymal bands",
            "sequela change",
            "sequela changes",
        ),
        "Pulmonary fibrotic sequela denotes chronic scar-like structural change in the lung.",
        "CT may show reticulation, traction bronchiectasis, architectural distortion, or bands.",
        ("right_lung", "left_lung", "lung_parenchyma"),
    ),
    LabelSpec(
        "pleural_effusion",
        "Pleural effusion",
        "Pleural effusion",
        "胸腔积液",
        ("pleural effusion", "pleural effusions", "pleural fluid"),
        "Pleural effusion is abnormal fluid in the pleural space.",
        "CT shows dependent pleural fluid, often with adjacent compressive atelectasis.",
        ("right_pleural_space", "left_pleural_space", "pleura"),
    ),
    LabelSpec(
        "mosaic_attenuation_pattern",
        "Mosaic attenuation pattern",
        "Mosaic attenuation pattern",
        "马赛克样密度",
        ("mosaic attenuation", "mosaic attenuation pattern", "mosaic perfusion"),
        "Mosaic attenuation is a patchwork of regions with differing lung attenuation.",
        "CT assessment considers small-airway, vascular, and infiltrative causes.",
        ("right_lung", "left_lung", "lung_parenchyma"),
    ),
    LabelSpec(
        "peribronchial_thickening",
        "Peribronchial thickening",
        "Peribronchial thickening",
        "支气管周围增厚",
        ("peribronchial thickening", "bronchial wall thickening", "thickening of the bronchial wall"),
        "Peribronchial thickening denotes increased thickness around bronchial walls.",
        "CT shows prominent or thickened bronchial walls, sometimes with adjacent inflammation.",
        ("tracheobronchial_tree", "right_lung", "left_lung"),
    ),
    LabelSpec(
        "consolidation",
        "Consolidation",
        "Consolidation",
        "肺实变",
        ("consolidation", "air-space opacity", "airspace opacity", "air bronchogram"),
        "Consolidation is replacement of alveolar air by fluid, cells, or other material.",
        "CT shows dense parenchymal opacity that may contain air bronchograms.",
        ("right_lung", "left_lung", "lung_lobes"),
    ),
    LabelSpec(
        "bronchiectasis",
        "Bronchiectasis",
        "Bronchiectasis",
        "支气管扩张",
        ("bronchiectasis", "bronchial dilatation", "dilated bronchus", "dilated bronchi"),
        "Bronchiectasis is irreversible abnormal dilatation of bronchi.",
        "CT signs include an increased bronchoarterial ratio and lack of normal bronchial tapering.",
        ("tracheobronchial_tree", "right_lung", "left_lung"),
    ),
    LabelSpec(
        "interlobular_septal_thickening",
        "Interlobular septal thickening",
        "Interlobular septal thickening",
        "小叶间隔增厚",
        ("interlobular septal thickening", "septal thickening", "thickened interlobular septa"),
        "Interlobular septal thickening is thickening of connective tissue outlining pulmonary lobules.",
        "CT may show smooth, nodular, or irregular linear opacities at lobular boundaries.",
        ("right_lung", "left_lung", "lung_parenchyma"),
    ),
)


LABEL_BY_ID = {spec.id: spec for spec in LABEL_SPECS}
LABEL_IDS = tuple(spec.id for spec in LABEL_SPECS)
SOURCE_COLUMN_TO_ID = {spec.source_column: spec.id for spec in LABEL_SPECS}
ID_TO_SOURCE_COLUMN = {spec.id: spec.source_column for spec in LABEL_SPECS}
LABEL_ZH = {spec.id: spec.zh for spec in LABEL_SPECS}


def require_label_id(label_id: str) -> str:
    if label_id not in LABEL_BY_ID:
        raise ValueError(f"Unsupported CT-RATE label: {label_id}")
    return label_id
