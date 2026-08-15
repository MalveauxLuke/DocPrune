Title: BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations

URL Source: https://arxiv.org/html/2501.03403

Markdown Content:
[1]\fnm Simone \sur Giovannini

\equalcont

These authors contributed equally to this work.

\equalcont

These authors contributed equally to this work.

\equalcont

These authors contributed equally to this work.

1]\orgdiv DINFO, \orgname Università degli Studi di Firenze, \orgaddress\street Via di Santa Marta, 3, \city Florence, \postcode 50139, \country Italy

2]\orgname LETXBE, \orgaddress\street 229 Rue Saint-Honoré, \city Paris, \postcode 75001, \country France

###### Abstract

We present a unified dataset for document Question-Answering (QA), which is obtained combining several public datasets related to Document AI and visually rich document understanding (VRDU). Our main contribution is twofold: on the one hand we reformulate existing Document AI tasks, such as Information Extraction (IE), into a Question-Answering task, making it a suitable resource for training and evaluating Large Language Models; on the other hand, we release the OCR of all the documents and include the exact position of the answer to be found in the document image as a bounding box. Using this dataset, we explore the impact of different prompting techniques (that might include bounding box information) on the performance of open-weight models, identifying the most effective approaches for document comprehension.

###### keywords:

Large Language Models (LLMs), Document AI, Dataset, Question Answering, Fine-tuning, Information Extraction.

††footnotetext: Accepted for publication in the International Journal of Document Analysis and Recognition (IJDAR).
1 Introduction
--------------

The increasing number of documents produced in various fields, including scientific research, legal proceedings, healthcare, and business, has created an enormous demand for efficient information extraction (IE) methods.

In document processing research, Optical Character Recognition (OCR) has proven essential for transforming scanned documents and images into machine-readable text, facilitating further analysis. Initially, statistical methods [mihalcea-tarau-2004-textrank] were used alongside OCR to extract information, followed by machine learning approaches. Subsequently, deep learning techniques [DBLP:journals/corr/abs-1810-04805], especially methods related to Natural Language Processing (NLP), became crucial in advancing document understanding. Today, the focus has shifted towards Large Language Models (LLMs) [DBLP:journals/corr/abs-2303-08774], which, with their exceptional ability to model natural language in complex contexts, have further enhanced document comprehension and the automation of information extraction from extensive volumes of text.

OCR tools and LLMs are now extensively used to perform several tasks in Document AI, including:

*   •Document Image Classification: classifies document images into types such as invoices, scientific papers, and receipts [docimageclass]; 
*   •Visual Information Extraction: extracts entities and relationships from unstructured content, considering text, visual elements, and layout [DBLP:conf/aaai/WangLJT0ZWWC21]; 
*   •Visual Question Answering: answers natural language questions based on a document’s content [docvqa]. 

The two main motivations for building the BoundingDocs dataset††The dataset is publicly available at [https://huggingface.co/datasets/letxbe/BoundingDocs](https://huggingface.co/datasets/letxbe/BoundingDocs)., that is focused on Information Extraction and Question Answering, are:

1.   1.the lack of extensive and diverse QA datasets in the field of Document AI; 
2.   2.the lack of precise spatial coordinates in the existing datasets. 

Current datasets do not effectively incorporate positional data, which is essential for reducing hallucinations and improving performance by enabling LLMs to understand document layout more precisely. In contrast, the proposed dataset, BoundingDocs, is specifically designed to capture positional information, which not only defines the dataset’s core feature but also serves multiple purposes. First, it allows for verifying whether a model correctly extracts both the value and its location, providing deeper insights into the model’s comprehension of document structure and helping to mitigate hallucinations. Second, it can be leveraged to enhance prompting by incorporating more detailed layout-aware instructions, further improving the model’s ability to interpret and generate structured outputs.

### Contribution

In this work, we propose a unified approach to build a Question-Answering dataset. Such a dataset can be used for evaluating how good Document AI models are to extract relevant information when answering to natural language questions. In doing so, we aim to address the following research questions:

*   •RQ1: How can existing datasets be unified into a common Question-Answering format? 
*   •RQ2: Can rephrased questions generated by LLMs enhance answer accuracy for document-based questions? 
*   •RQ3: Does including layout information in prompts (e.g. [DBLP:journals/corr/abs-2306-00526, DBLP:conf/icdar/LamottWUSKO24]) improve the model’s performance on document comprehension tasks? 

To explore these questions, our study is organized into the following sections. Section [2](https://arxiv.org/html/2501.03403v3#S2 "2 State of the art ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") reviews the existing literature and benchmarks in the field of Document AI and question answering tasks; Section [3](https://arxiv.org/html/2501.03403v3#S3 "3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") describes the process of unifying datasets into a common Question-Answering format with enhanced layout annotations; Section [4](https://arxiv.org/html/2501.03403v3#S4 "4 Experimental results ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") evaluates the performance of LLMs using various prompting techniques and presents the results. Conclusions are drawn in Section [5](https://arxiv.org/html/2501.03403v3#S5 "5 Conclusions ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") where we discuss our findings, the key challenges encountered, and propose directions for future research.

2 State of the art
------------------

We provide an overview of the main models and techniques proposed for Question Answering (QA) and Visual Question-Answering (VQA) [docvqa]. We also discuss the features of the main datasets in the Document AI that we considered in our research.

### 2.1 Related datasets

As summarized in Table[1](https://arxiv.org/html/2501.03403v3#S2.T1 "Table 1 ‣ 2.1 Related datasets ‣ 2 State of the art ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"), we selected datasets that best match our focus on comprehensive document understanding and advanced VQA, addressing challenges across both single-page and multi-page documents. One detailed review covering more datasets focused on layout-related tasks can be found in [gemelli2024].

Table 1: Datasets review details. For clarity, the following codes are used in the table: OCR info - 1: Full text with bboxes, 2: Partial text with bboxes, 3: Full text without bboxes; OCR engine - 0: Not specified, 1: Tesseract, 2: Amazon Textract, 3: Microsoft OCR, 4: Azure Cognitive Service, 5: Synthetic document (OCR not needed; text is pre-known); Type - 1: Real, 2: Synthetic; Lang - 1: English, 2: Italian, 3: French, 4: Spanish, 5: Chinese, 6: German, 7: Portuguese, 8: Japanese, 9: Indonesian, 10: Not specified mix.

Among the foundational datasets, DocVQA[docvqa, TITO2023109834] stands as one of the earliest benchmarks dedicated to VQA on document images, focusing on understanding both textual and layout aspects of documents. Introduced in 2020, DocVQA comprises multiple tasks designed to push the boundaries of document comprehension. The main tasks include answering questions about individual document pages and analyzing multi-page documents — a crucial capability for real-world applications. The Single Page[docvqa] subset includes 50,000 questions over 12,767 documents, while the Multi Page[TITO2023109834] subset contains 46,436 questions spanning 5,929 documents (covering 47,952 pages in total). These datasets require models to interpret the visual structure of documents and to derive insights that go beyond simple text extraction.

DUDE[dude] builds on this foundational work by extending VQA to multi-domain, multi-purpose documents. The dataset provides 5,000 annotated PDF files with 18,700 question-answer pairs across various domains and time frames, making it a unique resource for tasks that integrate Document Layout Analysis with complex, layout-based question answering. Unlike typical QA datasets, DUDE often requires multi-step reasoning, handling both content and structural queries. For instance, questions may include layout-based prompts such as “How many text columns are there?” or require arithmetic and comparison skills, presenting a challenging dataset for models trained primarily on text-based QA.

In addition to these datasets, several others serve as standard benchmarks and are worth mentioning briefly. VRDU[vrdu] includes two corpora—registration forms from the U.S. Department of Justice and ad-buy forms from the FCC—representing templates of varying complexity. The FATURA[fatura] dataset provides 10,000 images across 50 templates with imbalanced distributions for fields commonly found in invoices, such as buyer information and total amount, along with bounding box annotations for structured data extraction. Kleister[kleister] datasets offer specialized financial reports and legal documents, with Kleister Charity and Kleister NDA addressing entity extraction for key attributes. Deepform[deepform] offers approximately 20,000 labeled receipts for political ad purchases with labeled fields for specific political advertising details.

Finally, FUNSD[funsd] and XFUND[xfund] are form-centric datasets focused on entity linking and key-value extraction in noisy, often multilingual documents. FUNSD includes 199 annotated forms in English, designed for form understanding, while XFUND broadens this to a multilingual setting with documents in seven languages, capturing the diversity of form structures globally.

Recently, several unified datasets similar to BoundingDocs have been published, aiming to integrate multiple document sources or tasks sharing both similarities and differences with our approach. Docmatix[huggingface2024docmatix] and K2Q[DBLP:conf/emnlp/ZmigrodSSMNLV24] are Information Extraction datasets containing 2.1 million and 12,000 documents, respectively. Both generate question-answer pairs automatically using LLMs. Notably, K2Q, which is only available upon request, implements a similar strategy to ours by augmenting questions that were initially created using fixed templates at the dataset level. However, neither dataset includes information about the position of the answer within the text, which distinguishes BoundingDocs.

Additionally, recently published datasets such as BigDocs[DBLP:journals/corr/abs-2412-04626] and DocStruct4M[hu2024mplugdocowl] aim to aggregate multiple tasks beyond Information Extraction. These datasets contain 7 million multimodal documents and 4 million documents, respectively. While significantly larger than BoundingDocs, they fall outside our focus on business documents. Moreover, they encompass a wide range of tasks, including Screenshot2HTML, Table2LaTeX, ChartParsing, and TableParsing. Not all examples in these datasets follow a question-answering format, and, most notably, they do not provide bounding box information for the answers.

### 2.2 Related methods

In recent years, the QA task [docvqa, TITO2023109834] has been approached in many ways, leveraging different techniques and various model architectures. These methods can be broadly categorized into NLP-based, LLM-based, and multimodal architectures, each addressing different aspects of document understanding and question answering.

NLP-based approaches build on general Question-Answering models, primarily focusing on text semantics without explicitly incorporating document layout or visual features. A prime example is BertQA[docvqa], which utilizes a BERT architecture followed by a classification head to predict the start and end indices of an answer span. Modifications such as changes in hyperparameters and the introduction of new pre-training tasks have been explored in multiple works [Garncarek_2021, liu2019robertarobustlyoptimizedbert], resulting in improved outcomes.

LLM-based methods leverage large language models to perform document understanding tasks by encoding structural and layout information directly into the input. For instance, LMDX[perot2023lmdx] incorporates layout information via bounding box coordinates in the prompt, enhancing retrieval precision and reducing hallucinations. DocLLM[wang2023docllm], which builds on the LayoutLM family, includes a specialized pretraining phase focused on structured layout data to improve document layout understanding. In contrast, NuExtract[numind2024nuextract] is designed for extracting structured JSON data from documents, using training data derived from the Colossal Clean Crawled Corpus[dodge2021documentinglargewebtextcorpora].

Multimodal architectures combine visual and textual features to enhance document comprehension across layout, content, and structure. Among OCR-free methods, mPLUG-DocOWL 1.5[hu2024mplugdocowl] integrates a Vision Transformer (ViT) [dosovitskiy2021imageworth16x16words] with an LLM for comprehensive Document AI analysis, aligning layout and textual cues effectively without requiring separate OCR stages. Similarly, Donut[kim2022ocrfreedocumentunderstandingtransformer] and Dessurt[davis2022endtoenddocumentrecognitionunderstanding] operate without OCR preprocessing, directly integrating image and text data for robust document understanding.

In contrast, OCR-dependent models further refine document comprehension by incorporating OCR-based tokens. Hi-VT5[TITO2023109834], for example, combines OCR tokens with visual features, optimizing its effectiveness for Question-Answering tasks that rely on precise textual information. Additionally, LayoutLMv3[huang2022layoutlmv3pretrainingdocumentai] introduces visual patch embeddings in place of traditional CNNs to better align text, layout, and visual cues, resulting in improved performance on tasks requiring fine-grained structural interpretation.

3 Dataset construction
----------------------

![Image 1: Refer to caption](https://arxiv.org/html/2501.03403v3/imgs/diagram1-new.png)

Figure 1: Dataset construction pipeline. The process begins with two main dataset categories: QA datasets (red) and Key-Value extraction datasets (blue). Both categories are processed using AWS OCR (Textract), followed by annotation matching. For the Key-Value extraction datasets, questions are generated and rephrased using Mistral 7B v0.3. All processed components are then unified into BoundingDocs (purple).

We base our new dataset, BoundingDocs, on the following datasets selected from Table [1](https://arxiv.org/html/2501.03403v3#S2.T1 "Table 1 ‣ 2.1 Related datasets ‣ 2 State of the art ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"): SP-DocVQA, MP-DocVQA, DUDE, Deepform, VRDU, FATURA, Kleister Charity, Kleister NDA, FUNSD, and XFUND. Our selection focuses specifically on business document-based datasets, thereby excluding datasets such as InfographicsVQA[infographicvqa] and SlideVQA[DBLP:conf/aaai/TanakaNNHSS23], which primarily deal with non-business documents, as well as VisualMRC[DBLP:conf/aaai/TanakaNY21], which, despite being document-based, features abstractive rather than extractive answers, as detailed in Section [3.2.1](https://arxiv.org/html/2501.03403v3#S3.SS2.SSS1 "3.2.1 Dataset preparation ‣ 3.2 Producing annotations ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations").

Although most of the selected datasets originate from Key-Value extraction tasks, which may involve recurring question types, BoundingDocs ensures diversity by incorporating datasets with heterogeneous characteristics. Specifically, BoundingDocs includes documents with widely varying layouts, structural complexities, and languages, spanning multiple domains such as invoices, contracts, forms, receipts, and documents containing some handwritten-filled forms. This collection provides an essential resource for training and evaluating Document AI models.

In Figure [1](https://arxiv.org/html/2501.03403v3#S3.F1 "Figure 1 ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") we show the implemented pipeline for dataset construction.

### 3.1 Dataset format definition

For each document, a JSON file contains the annotation (examples in Figure [2](https://arxiv.org/html/2501.03403v3#S3.F2 "Figure 2 ‣ 3.2.2 Matching annotations and OCR ‣ 3.2 Producing annotations ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations")). Each word in the answer is linked to its corresponding bounding box. Following established practices in the literature (e.g., LayoutLM[huang2022layoutlmv3pretrainingdocumentai], BERT[DBLP:journals/corr/abs-1810-04805]), the bounding boxes are normalized integers ranging from 0 to 1000 relative to the actual page size. Each bounding box is defined by a list of four values: the width, the height, the X X and Y Y coordinates of the top-left vertex of the rectangle.

### 3.2 Producing annotations

A significant challenge comes from integrating various types of annotations into a unified structure. Datasets like Deepform, Kleister, and FATURA provide annotations that only establish a relationship between a key and its corresponding value in the text, such as annotating _Address = 48 Woodford, SandyFord_. However, these datasets lack essential positional information, such as the text’s location, frequency of occurrence, and page number. In contrast, datasets like VRDU and MP-DocVQA provide different types of positional information: the former provides bounding boxes for the values to be extracted, while the latter only specifies which page of the document contains the answer. However, inconsistencies may arise because these datasets utilize different OCR tools, leading to variations in positional measurements and formats. To ensure consistent calculations for bounding box positions, we selected Amazon Textract [amazon_textract], as it is both already deployed in our company’s document processing systems (reducing annotation costs and ensuring consistency) and one of the most widely used commercial OCR services.

In the case of FUNSD and XFUND, the datasets contain annotations related only to the text’s structure and relationships between elements. Consequently, additional steps are necessary to generate relevant questions from these datasets.

#### 3.2.1 Dataset preparation

Upon collecting and downloading the datasets the following preliminary operations have been considered case by case. These additional steps are critical to standardize and prepare the datasets for the generation of annotations. Note that when the test split is not public, as in Kleister, DUDE, and DocVQA, the documents in those test splits and their corresponding annotations are excluded from BoundingDocs.

Annotation Conversion: When the annotations in a dataset have a complex format, they are converted into a standardized, more straightforward format. This is particularly required for the VRDU dataset, where the original annotations require interpretation and conversion.

Filtering Pages/Questions: Some datasets contain redundant or irrelevant content, such as unnecessary pages or questions, which have been removed. For instance, in the DocVQA dataset, pages from the Multi Page set were excluded from the Single Page set to prevent duplication. Additionally, for both DUDE and DocVQA datasets, we filtered out all questions that can be defined as abstractive, meaning that the answer is not explicitly present in the document text but instead requires reasoning or synthesis. Since our objective is to provide the exact location of each answer within the document, it is essential to retain only extractive questions, where the answer can be directly found in the text. This ensures that every identified answer has a corresponding position in the document, which would not be possible for abstractive questions.

Downloading Original Documents: In datasets where only annotations are provided without the corresponding documents, the original documents are downloaded from external sources. This step was necessary for the Deepform dataset, where the PDFs were not included alongside the annotations.

OCR Processing with Textract: To ensure consistency across all datasets, Amazon Textract has been applied to all documents, regardless of whether they already contained OCR data. Datasets were processed through Textract not only when OCR data was completely absent, but also when OCR was only provided for the annotated fields. This process has been applied to datasets such as VRDU, FATURA, Kleister, SP-DocVQA, Deepform, FUNSD, and XFUND, where OCR data is either insufficient or not provided.

Key-Value Association Creation: For FUNSD and XFUND, key-value pairs for information extraction were generated from the annotations. This step involves linking elements labeled as questions to their corresponding answers to facilitate coherent information extraction.

#### 3.2.2 Matching annotations and OCR

To match the answer to each question with the data extracted by Textract [amazon_textract], a script has been developed whose main challenge is to identify the correct word when the same value appears at multiple positions. Our approach matches the annotated value with the extracted text and considers all occurrences as potential matches. While this method ensures broad coverage, it may lead to false positives when the same textual value appears in unrelated contexts. A thorough analysis of this aspect can be found in Section [3.3](https://arxiv.org/html/2501.03403v3#S3.SS3 "3.3 Statistics & splits ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations").

A considerable time has been devoted to produce high quality annotations. This script, a significant part of our contribution, is used across all datasets with only slight modifications to match the different annotation formats.

For a document and a given key-value pair, where the key represents a label (such as “name,” “address,” or “date”) describing the type of information, and the value contains the actual data associated with that label, the script executes the following steps (no brand new annotations are created):

Figure 2: Sample of QA pairs from the dataset. The left QA pair is sourced from Deepform, while the right one is from Kleister Charity. The purple values represent the specific details related to each QA pair, and the blue keys denote the fixed structure defined for our dataset.

1.   1.Compare each text line extracted by Textract (Line) with the correct answer using Jaccard similarity. The Jaccard similarity between two sets A A and B B is given by: J​(A,B)=|A∩B||A∪B|J(A,B)=\frac{|A\cap B|}{|A\cup B|} where |A∩B||A\cap B| is the number of common elements between the two sets, and |A∪B||A\cup B| is the total number of unique elements across both sets. 
2.   2.If similarity exceeds a given threshold, the Line is added to a set of candidates. 
3.   3.For each candidate line that exceeds the Jaccard similarity threshold, we verify that each word of the ground truth answer is also detected as a Word block by Textract and falls within the Line bounding box. The set of all Word blocks (and their corresponding bounding boxes) that, when concatenated, reconstruct the original answer becomes the localized and annotated answer in BoundingDocs. If the answer is found in multiple locations on the page through this procedure, all occurrences are considered valid answers. 

Note that all date annotations in the Kleister datasets follow the _’YYYY-MM-DD’_ format. If a date appears in a different format within the document, its original annotation is standardized to this format. Our script is designed to handle this specific case through the use of regular expressions. This was not necessary for the other datasets.

#### 3.2.3 Questions formulation

For datasets that are not originally designed for QA but only for key-value extraction, it is necessary, after matching the answers with the OCR output, to also generate the corresponding questions. Therefore, in addition to the three steps described in the previous section, the following two additional steps are performed:

1.   4.Questions are generated using the template What is the [key name]? (e.g., What is the Address?). Note that [key name] does not refer to the actual name assigned to the key by the datasets (e.g., program_desc) but rather to a refined, natural-language version defined by us (e.g., program description). We manually refined around 10 keys per dataset. This ensures that questions are always in natural language—though not necessarily correct—but consistently composed of meaningful words. For XFUND, the question template was automatically translated to match document languages. Datasets with pre-defined questions (DUDE, MP-DocVQA, SP-DocVQA) used their own questions. 
2.   5.

Moreover, for VRDU Ad Buy Form, additional questions are created to account for key-value pairs linked to specific ad programs, such as:

    *   •What is the [program_start_date] for [program_desc]? 
    *   •What is the [program_end_date] for [program_desc]? 
    *   •What is the [sub_amount] for [program_desc]? 

#### 3.2.4 Rephrasing questions

After the previously described steps, the questions for the new dataset are generated. Inspection of these questions, which followed a simple template-based structure, revealed that they are often grammatically incorrect, overly simplistic, and consistently adhered to the same pattern. This raised concerns that fine-tuning an LLM on these questions could introduce bias, potentially leading to poor performance on questions written by humans, which may not follow the template.

To mitigate this issue, we employed the Mistral 7B model [jiang2023mistral] to correct and rewrite the questions, aiming to fix errors and introduce linguistic diversity. Other Mistral models, such as Mistral Large[mistral2024large] and Mixtral 8x7B[mistral2023mixtral], were also tested, but they produced overly complex, verbose, and unnatural questions.

The prompt for question rewriting included manually written examples to guide the model, with no information about the correct answer to avoid biasing the generation. For example, the question What is the Gross Amount? was rewritten by the LLM as What is the value of the Gross Amount?.

This procedure was applied to most questions in the dataset, adding a new attribute, rephrased_question. Questions from DUDE, MP-DocVQA, and SP-DocVQA were excluded as they were already human-written.

To ensure quality and limit hallucinations, we iteratively refined the prompt during the design phase and validated the outputs through manual sampling. Additionally, the LLM was instructed to preserve the semantic meaning a from the template questions, using the original answer as a starting point.

Table 2: Overall statistics of BoundingDocs, divided by source.

In Figure [2](https://arxiv.org/html/2501.03403v3#S3.F2 "Figure 2 ‣ 3.2.2 Matching annotations and OCR ‣ 3.2 Producing annotations ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") we show one example of the final format of the dataset questions, including the rephrased version of the questions.

### 3.3 Statistics & splits

The dataset is split into training, validation, and test sets using an 80-10-10 split based on document count, ensuring that all questions related to a single document remain within the same set. Since the dataset comprises diverse sources with varying structures and question types, we propose our own split to ensure a balanced distribution of document layouts and question types across all sets. Table [2](https://arxiv.org/html/2501.03403v3#S3.T2 "Table 2 ‣ 3.2.4 Rephrasing questions ‣ 3.2 Producing annotations ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") provides an overview of the dataset’s size and source distribution, while detailed statistics can be found in the Supplementary Material.

To achieve this balance, documents from each source dataset are sampled separately. Specifically, documents from Deepform are split in an 80-10-10 ratio, followed by documents from FATURA, DUDE, and all other sources. The union of these individual splits yields the final training, validation, and test sets, ensuring that all sets reflect the dataset’s overall diversity.

Some of the pages annotated using the proposed algorithm and belonging to BoundingDocs are shown in Figures [3](https://arxiv.org/html/2501.03403v3#S3.F3 "Figure 3 ‣ 3.3 Statistics & splits ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") and [4](https://arxiv.org/html/2501.03403v3#S3.F4 "Figure 4 ‣ 3.3 Statistics & splits ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"). For illustration purpose, colored rectangles are drawn around the fields corresponding to the correct answers to the questions.

![Image 2: Refer to caption](https://arxiv.org/html/2501.03403v3/imgs/deepform_screenshot.png)

Figure 3: Deepform page with bbox annotations.

![Image 3: Refer to caption](https://arxiv.org/html/2501.03403v3/imgs/vrdu_registration_screenshot.png)

Figure 4: VRDU Registration Form page with bbox annotations.

It should be noted, as previously mentioned, that for datasets such as Kleister, DUDE, and DocVQA, the test splits are not publicly available. Consequently, documents from these test splits are not included in our dataset. For datasets like FATURA and Deepform, which do not have predefined splits, this issue does not arise. However, further discussion is needed regarding FUNSD, XFUND, and VRDU, as they do have defined splits. In the case of FUNSD and XFUND, a portion of the documents from their test splits were included in the random process used to define the splits in BoundingDocs. As a result, some of these documents are part of the BoundingDocs training split. Specifically, out of the 38,515 38,515 documents in the training split, 313 313 originate from the FUNSD/XFUND test sets, accounting for approximately 0.8%0.8\%. Regarding VRDU, its repository provides multiple splits depending on the task of interest, meaning there is no single definitive test split. Therefore, VRDU has been treated similarly to FATURA and Deepform. This analysis informs users that incorporating BoundingDocs into training data does not prevent models from being evaluated on benchmarks like DUDE, DocVQA, or Kleister. This leads to the conclusion that BoundingDocs can serve a dual purpose: it can be used as a pretraining tool without significant risk of contamination with established benchmarks, allowing these benchmarks to be employed for evaluating the resulting models. Additionally, BoundingDocs can function as a benchmark itself for assessing models in a question answering setting, even for datasets and document types that were not originally designed for this task.

### 3.4 Annotation assessment

To provide insights into the dataset quality, we analyzed the frequency of multiple matches within a single page, as mentioned in Section [3.2.2](https://arxiv.org/html/2501.03403v3#S3.SS2.SSS2 "3.2.2 Matching annotations and OCR ‣ 3.2 Producing annotations ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"). We found that in approximately 20% of cases, the annotated value appeared more than once on the same page. However, this does not necessarily indicate that 20% of the dataset contains false positives. To further investigate this phenomenon, we conducted a manual audit of annotations with multiple matches. We randomly sampled 20 pages containing at least one question with multiple valid answers, selecting one such question per page. We successfully sampled 20 pages for most datasets, with the exception of FUNSD (19 pages), SP-DocVQA (13 pages), and FATURA and VRDU Registration Form (0 pages for both). For each sampled page-question pair, we manually examined all annotated answer bounding boxes to verify whether all instances referred to the same logical answer or if some were unrelated occurrences of the same textual value. An annotation was considered correct only if all annotated answers corresponded to valid responses to the question, rather than being semantically unrelated occurrences of the same text. The results, reported in Table [3](https://arxiv.org/html/2501.03403v3#S3.T3 "Table 3 ‣ 3.4 Annotation assessment ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"), reveal that approximately 38% of questions with multiple answers per page contain at least one answer that is not logically related to the question. Extrapolating from this finding, we estimate that approximately 7% of all annotations (38% of the 20% with multiple matches) may exhibit this issue. Examining the results by data type, we observe distinct patterns in annotation accuracy. Acronyms achieve perfect accuracy (100%), as they are highly specific and rarely repeated for different purposes within documents. Named entities and currency values show moderate accuracy (68.12% and 70.00%, respectively), while dates perform slightly lower at 66.67%. Numbers exhibit the lowest accuracy rate (46.88%), as numerical values frequently appear in unrelated sections of documents such as page numbers, reference codes, or unrelated quantities. The “Other” category, which includes miscellaneous data types, shows an accuracy of 40.00%, indicating challenges in handling diverse or less structured information.

Table 3: Annotation quality statistics by data type for questions with multiple valid answers per page.

### 3.5 Dataset examples

Table 4: QA pairs of the examples in Figure [3](https://arxiv.org/html/2501.03403v3#S3.F3 "Figure 3 ‣ 3.3 Statistics & splits ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") and Figure [4](https://arxiv.org/html/2501.03403v3#S3.F4 "Figure 4 ‣ 3.3 Statistics & splits ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"). The first two refer to the Deepform sample and the last two to the VRDU Registration Form one.

![Image 4: Refer to caption](https://arxiv.org/html/2501.03403v3/imgs/diagram2-new.png)

Figure 5: Experimental framework showing the different prompting strategies implemented for vision and text language models. Vision LLMs (blue) are prompted with images and rephrased questions, while text LLMs (red) are prompted with three different configurations of page content and questions. Some models underwent supervised fine-tuning before testing, while others were evaluated directly.

In Fig. [3](https://arxiv.org/html/2501.03403v3#S3.F3 "Figure 3 ‣ 3.3 Statistics & splits ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") (Deepform) and Fig. [4](https://arxiv.org/html/2501.03403v3#S3.F4 "Figure 4 ‣ 3.3 Statistics & splits ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") (VRDU Registration Form) it is possible to analyze two pages the corresponding QA pairs are shown in Table [4](https://arxiv.org/html/2501.03403v3#S3.T4 "Table 4 ‣ 3.5 Dataset examples ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"). In Fig. [3](https://arxiv.org/html/2501.03403v3#S3.F3 "Figure 3 ‣ 3.3 Statistics & splits ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") the extracted fields are the advertiser’s name and the gross amount for the various transmissions. In Fig. [4](https://arxiv.org/html/2501.03403v3#S3.F4 "Figure 4 ‣ 3.3 Statistics & splits ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") the fields to be extracted are only two: the registrant name and the registration number.

These two examples illustrate how, despite the large number of documents in the collection, the potential amount of information present in the documents is underutilized, as the annotated fields are few compared to the entire body of the documents, indicating that the potential of this large document collection is not being properly exploited.

To provide a concrete response to the issue raised in Section [3.2.2](https://arxiv.org/html/2501.03403v3#S3.SS2.SSS2 "3.2.2 Matching annotations and OCR ‣ 3.2 Producing annotations ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"), as shown in Fig. [3](https://arxiv.org/html/2501.03403v3#S3.F3 "Figure 3 ‣ 3.3 Statistics & splits ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"), the value $119,000.00 has been detected and annotated four times on the page. The two entries on the left part of the page are certainly correct, as they refer to totals: the first is related to the date and the second to the TV transmission. Regarding the values stacked on the right side of the page, the value beneath represents the sum of all those above it, and in this case, the numbers match. This validates the annotation, as the total refers to the same entity. If there were additional entries, only the total would be annotated, as the partial values would necessarily differ.

Additional examples that provide a full overview of the entire variety of the dataset can be found in the Supplementary Material.

4 Experimental results
----------------------

Tables [5](https://arxiv.org/html/2501.03403v3#S4.T5 "Table 5 ‣ 4.3 Baseline models ‣ 4 Experimental results ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"), [6](https://arxiv.org/html/2501.03403v3#S4.T6 "Table 6 ‣ 4.4 Ablation study: question formulation ‣ 4 Experimental results ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"), and [7](https://arxiv.org/html/2501.03403v3#S4.T7 "Table 7 ‣ 4.6 Fine-Tunings ‣ 4 Experimental results ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") present our experimental results across the different datasets and model configurations. The fine-tuning and testing pipeline implemented is summarized and plotted in Figure [5](https://arxiv.org/html/2501.03403v3#S3.F5 "Figure 5 ‣ 3.5 Dataset examples ‣ 3 Dataset construction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"), which depicts the various types of prompts used depending on the model type: for vision models, we provided the image along with the question only, while for traditional LLMs, we used the OCR-extracted page text, the question, and, for some ablation, the bounding boxes of the text. Notably, all the fine-tuning experiments described were conducted using only 10% of the training split and evaluated on the proposed test split.

The three tables follow the same format, where the columns correspond to the documents in the BoundingDocs test set derived from the respective dataset. All models and configurations have been evaluated on the proposed test split. It is crucial to emphasize that these results are not directly comparable to established benchmarks (e.g., DocVQA or DUDE) or to performances reported in other studies using the same datasets. Additionally, comparisons between different model types (e.g., LLMs vs. vision LLMs) should be avoided due to differences in prompting techniques and LoRA fine-tuning parameters (details in the Supplementary Material). The test split is specifically determined by our question-filtering strategy and dataset construction methodology. Therefore, these results primarily serve to illustrate the dataset’s difficulty and provide a baseline, rather than to position our approach relative to the state of the art.

### 4.1 Evaluation metrics

We use the standard metric ANLS* [peer2024anlsuniversaldocument], which supports a wider range of tasks including line-item extraction and document-processing tasks.

For each model-dataset pair in our results, we report two key measurements: the ANLS* value (rescaled between 0 and 100 for easier reading) and the percentage of non-JSON parsable responses relative to the total number of queries. The weighted average provides a comprehensive overview based on the number of examples for each dataset. For ANLS*, higher values indicate better performance, while for non-parsable responses, lower percentages are preferable. In the tables, the best values for each dataset are in bold, whike the second-best values are underlined.

### 4.2 Prompt construction

In this study, each question in the dataset may have answers distributed across multiple pages. Processing multi-page documents poses significant computational challenges, as noted in works such as Multi-PageDocVQA [TITO2023109834]. Additionally, the context size limitations of smaller LLMs make encoding all pages into a single prompt impractical. To address these constraints, we adopt an atomic approach, processing each relevant page separately rather than constructing a single comprehensive prompt.

Our experiments follow the ‘oracle’ setup described in [TITO2023109834], where only the page containing the answer is provided as input to the model. This setup isolates the model’s answering capabilities from variations in input sequence length, serving as a theoretical upper bound on performance—assuming the method correctly identifies the relevant page. As a result, these baselines reflect the intrinsic difficulty of the questions and document layouts rather than external factors such as content filtering. Nevertheless, the dataset includes complete documents, allowing users to implement and evaluate multi-page methods, which must also handle the task of identifying relevant pages. For questions requiring information from multiple pages, we generate independent prompts for each relevant page, appending the same question to each. For instance, if a five-page document contains relevant information on pages 2 and 4, we create two separate prompts: one incorporating content in page 2 and the other incorporating page 4, both paired with the same question.

Each prompt consists of three components: the document text, the question, and a given answer format (JSON) to facilitate structured data extraction.

### 4.3 Baseline models

Table 5: Baseline models. ANLS* scores and JSON parsing error percentages across datasets for baseline models on our test split. ANLS* scores measure accuracy in answering document questions, while the bottom value in each cell shows JSON parsing errors, indicating output consistency. The ”W. Avg” column provides a weighted average across datasets, with bold and underlined values marking the top two scores per dataset.

We evaluated three popular open-weight models as baselines: Mistral 7B Instruct v0.3[jiang2023mistral], Llama 3 8B Instruct[touvron2023llama], and Phi 3.5 3.8B Instruct[abdin2024phi3]. These models were chosen for their established performance and recognition in the NLP community.

In addition to these text-only models, we also tested three multimodal models capable of processing both text and images: Claude 3.7 Sonnet[claudesonnet] and Qwen2-VL[DBLP:journals/corr/abs-2409-12191] in its 2B and 7B variants. This selection ensures a diverse range of model types and sizes, allowing for a comprehensive evaluation. These tests served to establish an initial benchmark and are reported in Table [5](https://arxiv.org/html/2501.03403v3#S4.T5 "Table 5 ‣ 4.3 Baseline models ‣ 4 Experimental results ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations").

Through a qualitative analysis based on the results of our preliminary experiments and observations of dataset examples, we identified the most complex layouts and the sources for which the proposed task is particularly challenging. Notably, Deepform and VRDU Ad Buy Form contain numerous similar values that can correspond to different entries in the financial report, complicating information extraction. Despite its seemingly simple layout, Kleister NDA also presents significant challenges. This suggests that the lack of a clear structure — characteristic of contract pages with dense and compact text — impedes the models’ ability to retrieve highly specific information. A comprehensive overview of the various sources that compose BoundingDocs can be found in the Supplementary Material.

### 4.4 Ablation study: question formulation

Table 6: Ablation study. ANLS* scores and JSON parsing error percentages across datasets for each prompting configuration on our test split. ANLS* scores measure accuracy in answering document questions, while the bottom value in each cell shows JSON parsing errors, indicating output consistency. The ”W. Avg” column provides a weighted average across datasets, with bold and underlined values marking the top two scores per dataset.

For investigating the impact of question formulation, we selected the Mistral 7B v0.3 (base version) for fine-tuning (details in the Supplementary Material). Results are reported in Table [6](https://arxiv.org/html/2501.03403v3#S4.T6 "Table 6 ‣ 4.4 Ablation study: question formulation ‣ 4 Experimental results ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"). We evaluated two types of questions—template-based (simple, consistent format) and rephrased (more varied, user-friendly language). Each model was tested with both question types, resulting in four experimental conditions:

*   •Template-Template: Model trained and tested with template-based questions. 
*   •Template-Rephrased: Model trained with template-based questions, tested with rephrased questions. 
*   •Rephrased-Template: Model trained with rephrased questions, tested with template-based questions. 
*   •Rephrased-Rephrased: Model trained and tested with rephrased questions. 

### 4.5 Incorporating bounding box information

To assess the impact of spatial information, we incorporated bounding box coordinates into the prompts, denoted as Reph.-Reph.-bbox in Table [6](https://arxiv.org/html/2501.03403v3#S4.T6 "Table 6 ‣ 4.4 Ablation study: question formulation ‣ 4 Experimental results ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"). Each Textract-extracted text element in the prompt was annotated with bounding box coordinates, enabling the model to reference spatial context. In this configuration, the model was specifically fine-tuned to produce more complex JSON outputs that include not only the answer, but also a comprehensive list of all locations where the extracted value appears in the document. While this approach provided richer spatial awareness, the requirement to generate more structured outputs introduced additional complexity that led to increased parsing errors.

To address these parsing challenges, we implemented the Reph.-Reph.-bbox w/regex configuration, which introduced a regex-based post-processing step. When the model’s structured JSON output was not parsable due to format inconsistencies or generation errors, the regex extraction mechanism served as a fallback solution to retrieve the target value, effectively maintaining the benefits of spatial information while mitigating the impact of parsing failures.

### 4.6 Fine-Tunings

Table [7](https://arxiv.org/html/2501.03403v3#S4.T7 "Table 7 ‣ 4.6 Fine-Tunings ‣ 4 Experimental results ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") presents the fine-tuning results for a subset of the models listed in Table [5](https://arxiv.org/html/2501.03403v3#S4.T5 "Table 5 ‣ 4.3 Baseline models ‣ 4 Experimental results ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"). In these experiments, models were trained and tested using rephrased questions. For Mistral, the prompt included both the page content and bounding boxes (corresponding to the Reph.-Reph.-bbox row in the Table [6](https://arxiv.org/html/2501.03403v3#S4.T6 "Table 6 ‣ 4.4 Ablation study: question formulation ‣ 4 Experimental results ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations")), whereas the two Qwen models received only the page image and the rephrased question as input. It is important to noice that these results are not directly comparable across models, as the fine-tuning settings for Mistral and Qwen differ significantly (details on fine-tuning settings can be found in the Supplementary Material.) Once again, these results are not intended for direct comparison, but rather to provide a baseline for the expected performance on this dataset after a fine-tuning of various models.

Table 7: Fine-tunings. ANLS* scores and JSON parsing error percentages across datasets for each fine-tuned model on our test split. ANLS* scores measure accuracy in answering document questions, while the bottom value in each cell shows JSON parsing errors, indicating output consistency. The ”W. Avg” column provides a weighted average across datasets, with bold and underlined values marking the top two scores per dataset.

### 4.7 Answer localization

In this article, we investigate the unique feature of BoundingDocs, bounding boxes, specifically to evaluate whether their inclusion can enhance document understanding. This can be achieved either by incorporating bounding box information as an additional input in textual prompts for off-the-shelf models, or by leveraging it for fine-tuning. Additionally, BoundingDocs can serve as a benchmark for assessing a model’s ability to accurately predict the location of an answer within a page. Our experiments, reported in [chen2025reliableinterpretabledocumentquestion], show that Claude 4 Sonnet achieves an average IoU of 0.031 on the BoundingDocs test set, while Qwen2.5-VL-7B reaches 0.048. These results demonstrate that, although off-the-shelf models perform reasonably well in retrieving information in a zero-shot setting, their capacity to indicate the precise location of the answer is effectively nonexistent, which significantly undermines the interpretability of their responses.

### 4.8 Research question answers

Our experimental findings provide clear answers to our research questions, defined in Section [1](https://arxiv.org/html/2501.03403v3#S1.SSx1 "Contribution ‣ 1 Introduction ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"):

*   •RQ1 - Dataset Unification: By standardizing data from various sources (e.g., receipts, invoices, forms) into a consistent Question-Answering format, models are exposed to a wide range of document layouts and content types, enhancing their training efficiency. This unification significantly streamlines the fine-tuning process, making it easier to handle diverse document sources. Moreover, fine-tuned models show significant improvements over instruct models, quantitatively confirming that exposure to varied document formats and layouts enhances the model’s ability to extract information, as can be seen comparing the overall results in Table [5](https://arxiv.org/html/2501.03403v3#S4.T5 "Table 5 ‣ 4.3 Baseline models ‣ 4 Experimental results ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") and Table [7](https://arxiv.org/html/2501.03403v3#S4.T7 "Table 7 ‣ 4.6 Fine-Tunings ‣ 4 Experimental results ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"). 
*   •RQ2 - Question Formulation Impact: The study revealed significant insights into how different question formulation strategies affect document comprehension and information extraction. As reported in Table [6](https://arxiv.org/html/2501.03403v3#S4.T6 "Table 6 ‣ 4.4 Ablation study: question formulation ‣ 4 Experimental results ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"), the Template-Template configuration demonstrated superior performance by leveraging structured, consistent question patterns. However, the Rephrased-Rephrased configuration emerged as a particularly robust solution, maintaining high ANLS* scores (e.g., 99.8 on FATURA, 96.4 on VRDU-Ad) while achieving 0% parsing errors across most datasets. Crucially, unlike other configurations, its performance remained stable regardless of whether the input followed a template or an augmented version, indicating that the model had developed a greater adaptability to diverse question structures. Notably, the Template-Rephrased setup performed least effectively, highlighting the challenges in transitioning from template-trained models to complex question structures. 
*   •RQ3 - Layout Information: The incorporation of spatial information in prompts yielded measurable improvements in model performance, as reported in Table [7](https://arxiv.org/html/2501.03403v3#S4.T7 "Table 7 ‣ 4.6 Fine-Tunings ‣ 4 Experimental results ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"). The Reph-Reph-bbox configuration achieved the highest weighted average ANLS* (91.6) across all datasets, demonstrating consistent improvements over configurations without spatial information. Notable gains were observed in complex document understanding tasks, with ANLS* scores increasing to 71.2 on XFUND and 83.0 on SP-VQA. While the initial implementation showed increased parsing errors, the addition of regex-based post-processing (Reph-Reph-bbox w/regex) successfully maintained high performance while reducing error rates to competitive levels (1.53% weighted average). 

5 Conclusions
-------------

The paper addresses the growing need for evaluating LLMs in Document AI tasks by proposing a unified dataset designed for document QA taking into account the position of answers’ text in the document. Baseline experiments using off-the-shelf LLMs demonstrate the challenges of applying generic models to specialized Document AI tasks; the performance of instruct models reveal clear limitations in generating correct and well-structured answers.

The paper reveals that while off-the-shelf LLMs struggle with document-specific tasks, targeted fine-tuning can significantly improve their capabilities. Rephrasing questions using LLMs improves the models’ understanding and response accuracy across different question formulations, suggesting that LLMs benefit from exposure to diverse linguistic variations during training. Incorporating layout and positional information into the prompt led to improved accuracy across most datasets, but at the cost of a higher percentage of non-parsable responses, reflecting the increased complexity of generating JSON outputs that include bounding box information.

In future work, we aim to explore a wider range of prompting techniques tailored to different model families. Among the many types of information available in BoundingDocs, a key challenge is determining the optimal combination for effective information extraction. Open questions include whether images can fully replace textual content in prompts, whether bounding boxes remain essential even when using images, and how different modalities interact to enhance model performance. The constructed dataset and the experimental results provide a solid foundation for future research in Document QA. Fine-tuning models with enriched prompts has shown promising improvements.

Declarations
------------

Competing interests The authors declare no competing interests.

Appendix A Dataset statistics
-----------------------------

We now provide a quantitative illustration using tables and graphs to show the nature of the dataset in all its aspects.

Figure [6](https://arxiv.org/html/2501.03403v3#A1.F6 "Figure 6 ‣ Appendix A Dataset statistics ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") provide an overview of the dataset construction, showing how many documents and related questions from the various source datasets contribute to the overall dataset.

There are already several aspects to consider: first of all, it can be seen that Deepform is the dataset that contributes the most documents, but it has an average of about 2 questions per document, whereas the dataset that contributes the most questions is FATURA, with an average of more than 10 questions per document. Note that VRDU Ad Buy Form is the dataset that contains the most annotated fields, and both this aspect and the construction of additional questions for this particular dataset lead to a very high number of questions compared to the relatively low number of documents (an average of more than 34 questions per document). Also, note that there are very few documents related to SP-DocVQA: this is because, as already mentioned, most of the documents in this dataset were already present in MP-DocVQA, and there was no point in including them twice.

![Image 5: Refer to caption](https://arxiv.org/html/2501.03403v3/supplementary_img/2_documents_distribution.png)

Figure 6: Documents distribution across datasets

In Table [8](https://arxiv.org/html/2501.03403v3#A1.T8 "Table 8 ‣ Appendix A Dataset statistics ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") you can observe the distribution of the languages in which the questions in the dataset are posed. The only questions, along with their respective documents, that are not in English are those formulated on XFUND, and thus they represent a clear minority compared to the total count.

Table 8: Language distribution

After providing a general overview of the dataset’s composition, it is also interesting to conduct an analysis of the types of questions that were generated and which field were extracted by running the matching algorithm on the various source datasets. Obviously, this analysis can only be conducted on the original datasets that pertain to key value extraction, as the questions are constructed according to the previously described template. For datasets such as DUDE and DocVQA, it is not possible to perform this type of tracking.

For Deepform, there are only five fields for which questions have been constructed, as can be observed in Table [9](https://arxiv.org/html/2501.03403v3#A1.T9 "Table 9 ‣ Appendix A Dataset statistics ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"). The main fields present are the total cost incurred for the advertisement and the name of the advertiser. The fields Flight From and Flight To are date values that represent the start and end days of the spot’s transmission.

Table 9: Question distribution for Deepform dataset

Regarding FATURA, the range of extracted fields is much broader compared to the previous Deepform, as visible in Table [10](https://arxiv.org/html/2501.03403v3#A1.T10 "Table 10 ‣ Appendix A Dataset statistics ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"). With 50 different layouts within FATURA, not all documents contain the same fields, which is the reason for the significant differences in frequencies among some fields. It is notable that the field with the most questions is Date of purchase (9800), while the least frequent is Total amount to be paid (685).

Table 10: Question distribution for FATURA dataset

Table 11: Question distribution for Kleister Charity dataset

Regarding the Kleister Charity dataset, the range of extracted fields is relatively narrow compared to the FATURA dataset. As shown in Table [11](https://arxiv.org/html/2501.03403v3#A1.T11 "Table 11 ‣ Appendix A Dataset statistics ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"), the dataset primarily focuses on extracting information such as the Charity Name, Charity Number, Address Post Town, Address Postcode, and Address Street Line. The fields with the fewest questions are Spending Annually in British Pounds and Income Annually in British Pounds, indicating that financial details are less frequently extracted from this dataset.

The Kleister NDA dataset, as detailed in Table [12](https://arxiv.org/html/2501.03403v3#A1.T12 "Table 12 ‣ Appendix A Dataset statistics ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"), contains questions across a very limited set of fields: Jurisdiction, Party, Term, and Effective Date. The field with the most questions is Jurisdiction, followed by Party, while the Effective Date field has only a single question.

Table 12: Question distribution for Kleister NDA dataset

The VRDU Ad Buy Form dataset, as shown in Table [13](https://arxiv.org/html/2501.03403v3#A1.T13 "Table 13 ‣ Appendix A Dataset statistics ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"), contains a broader range of fields compared to the previous datasets. The fields with the most questions are Program Start Date, Channel, and Program End Date. In contrast, the field with the fewest questions is Program Description, with only 6 questions.

Table 13: Question Distribution for VRDU Ad Buy Form dataset

The VRDU Registration Form dataset, as detailed in Table [14](https://arxiv.org/html/2501.03403v3#A1.T14 "Table 14 ‣ Appendix A Dataset statistics ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"), contains questions across 6 different fields. The fields with the most questions are Registration Number and Registrant Name, while the field with the fewest questions is Signer Title.

Table 14: Question distribution for VRDU Registration Form dataset

The latest statistics worth noting are those related to the split made for training, validation, and testing. As previously described, the documents were divided according to an 80-10-10 percentage, assuming that the distribution of questions would be similar and that we would therefore obtain the same percentage division for the latter as well. As can be seen from the Table [15](https://arxiv.org/html/2501.03403v3#A1.T15 "Table 15 ‣ Appendix A Dataset statistics ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"), our intuition was confirmed, achieving the desired partitioning for the questions as well.

Table 15: Train/Val/Test split

Appendix B Questions rephrasing
-------------------------------

As outlined in the main text, a key method for generating BoundingDocs is question rewriting using LLMs, specifically Mistral 7B. The model was employed through a few-shot approach, as illustrated in Figure [7](https://arxiv.org/html/2501.03403v3#A2.F7 "Figure 7 ‣ Appendix B Questions rephrasing ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations"), where specific examples of the desired question reformulations were provided. The prompt did not include the answer to the question, as we observed that incorporating the answer in the few-shot examples influenced the question formulation. This led to the generation of questions biased toward facilitating information retrieval, thereby compromising the neutrality of the rewriting process.

Figure 7: Prompt example and LLM’s answer for questions rephrasing.

Appendix C Finetuning details
-----------------------------

For our work, we fine-tuned three different models: Mistral-7B-v0.3, Qwen2-VL-2B, and Qwen2-VL-7B. All three models were fine-tuned on the 10% of the train split using QLoRA, but with two different configurations: one for Mistral, which processes only text input, and another for the Qwen2-VL models, which also take document images as input. The fine-tuning process was performed in Python using the HuggingFace SFT trainer. Quantization was achieved using the BitsAndBytes library, and LoRA was applied through the peft, both of which are also Python libraries.

Table 16: Fine-tuning configurations for the Mistral and Qwen2-VL models.

Appendix D Dataset examples
---------------------------

Similarly to what was done in the paper, a comprehensive qualitative overview of the entire variety of the dataset will be provided. An example will be shown for each source dataset, along with (almost) all the corresponding QA pairs formulated for that page, as shown in Table LABEL:tab:app_samples.

Table 17: QA pairs of the examples, each pair referencing the specific example it corresponds to.

| Template Question | Rephrased Question | Answer | Fig. |
| --- | --- | --- | --- |
| What is Advertiser? | Who is the advertiser? | Jordan, Jonathan | [8](https://arxiv.org/html/2501.03403v3#A4.F8 "Figure 8 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Gross Amount? | What is the value of the Gross Amount? | $10,500.00 | [8](https://arxiv.org/html/2501.03403v3#A4.F8 "Figure 8 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Address Post Town? | What is the post town of the address? | Stoke-on-Trent | [9](https://arxiv.org/html/2501.03403v3#A4.F9 "Figure 9 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Address Postcode? | What is the postal code of the address? | ST4 8AW | [9](https://arxiv.org/html/2501.03403v3#A4.F9 "Figure 9 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Address Street Line? | What is the value of the Address Street Line? | 28 Greenway | [9](https://arxiv.org/html/2501.03403v3#A4.F9 "Figure 9 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Charity Name? | What is the name of the charity? | Lucas’ Legacy - Childhood Brain Tumour Research | [9](https://arxiv.org/html/2501.03403v3#A4.F9 "Figure 9 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Charity Number? | What is the charity number? | 1167650 | [9](https://arxiv.org/html/2501.03403v3#A4.F9 "Figure 9 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Jurisdiction? | In which state is the company registered? | Delaware | [10](https://arxiv.org/html/2501.03403v3#A4.F10 "Figure 10 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Party? | What is the name of the company? | Cisco Systems, Inc., | [10](https://arxiv.org/html/2501.03403v3#A4.F10 "Figure 10 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Contract ID? | What is the contract ID number? | 711207 | [11](https://arxiv.org/html/2501.03403v3#A4.F11 "Figure 11 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is the Product? | What is the name of the product being advertised? | Q42020 Broadcast | [11](https://arxiv.org/html/2501.03403v3#A4.F11 "Figure 11 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Property? | What is the property name? | KXLF | [11](https://arxiv.org/html/2501.03403v3#A4.F11 "Figure 11 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Agency? | Who is the advertising agency? | Left Hook Communications | [11](https://arxiv.org/html/2501.03403v3#A4.F11 "Figure 11 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Advertiser? | Who is the advertiser? | Bennett/Democrat/ Secretary of State | [11](https://arxiv.org/html/2501.03403v3#A4.F11 "Figure 11 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Gross Amount? | What is the value of the gross amount? | $3,020.00 | [11](https://arxiv.org/html/2501.03403v3#A4.F11 "Figure 11 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Sub Amount for M-F 530-7am News M-F 530-7am News? | What is the value for the ’Sub Amount’ key for ’M-F 530-7am News’? | $100.00 | [11](https://arxiv.org/html/2501.03403v3#A4.F11 "Figure 11 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is the Channel for M-F 530-7am News M-F 530-7am News? | What is the value of the ’Channel’ for the ’530-7am News’ broadcasted from Monday to Friday? | All | [11](https://arxiv.org/html/2501.03403v3#A4.F11 "Figure 11 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Program Start Date for M-F 530-7am News M-F 530-7am News? | What is the start date for the M-F 530-7am News program? | 10/06/20 | [11](https://arxiv.org/html/2501.03403v3#A4.F11 "Figure 11 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is the Program End Date for M-F 530-7am News M-F 530-7am News? | What is the end date for the program ’M-F 530-7am News’? | 10/12/20 | [11](https://arxiv.org/html/2501.03403v3#A4.F11 "Figure 11 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Registrant Name? | What is the name of the registrant? | KOREA TRADE PROMOTION CENTER | [12](https://arxiv.org/html/2501.03403v3#A4.F12 "Figure 12 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Registration Number? | What is the registration number for the company? | 1619 | [12](https://arxiv.org/html/2501.03403v3#A4.F12 "Figure 12 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Signer Title? | What is the signer’s title? | DEPUTY DIRECTOR | [12](https://arxiv.org/html/2501.03403v3#A4.F12 "Figure 12 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| First bubble in the HPA Axis? | First bubble in the HPA Axis? | Hypothalamus | [13](https://arxiv.org/html/2501.03403v3#A4.F13 "Figure 13 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What does CORT stand for in this document? | What does CORT stand for in this document? | Cortisol | [13](https://arxiv.org/html/2501.03403v3#A4.F13 "Figure 13 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| Where does cortisol go after it is sent from the adrenal cortex? | Where does cortisol go after it is sent from the adrenal cortex? | Hypothalamus | [13](https://arxiv.org/html/2501.03403v3#A4.F13 "Figure 13 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Buyer information? | What is the name of the buyer? | Buyer :Nichole Harrington 8282 Kristie Lights South Loriburgh, PR 35228 US Tel:+(227)782-8066 Email:blackjames@ example.net Site:http://ruiz-bailey.com/ | [14](https://arxiv.org/html/2501.03403v3#A4.F14 "Figure 14 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Date of purchase? | When was the purchase date? | Invoice Date: 30-Oct-1998 | [14](https://arxiv.org/html/2501.03403v3#A4.F14 "Figure 14 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Due date? | What is the due date? | Due Date : 24-May-2020 | [14](https://arxiv.org/html/2501.03403v3#A4.F14 "Figure 14 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Purchase order number? | What is the purchase order number value? | PO Number :72 | [14](https://arxiv.org/html/2501.03403v3#A4.F14 "Figure 14 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Seller Address? | What is the seller’s address? | Address:05866 Velazquez Mount North Diane, NJ 20651 US | [14](https://arxiv.org/html/2501.03403v3#A4.F14 "Figure 14 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Total amount before tax and discount? | What is the value of the total amount before tax and discount? | SUB_TOTAL : 293.47 $ | [14](https://arxiv.org/html/2501.03403v3#A4.F14 "Figure 14 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Tax? | What is the tax amount? | TAX:VAT (5.69%): 16.70 $ | [14](https://arxiv.org/html/2501.03403v3#A4.F14 "Figure 14 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Title? | What is the key for the title information? | TAX INVOICE | [14](https://arxiv.org/html/2501.03403v3#A4.F14 "Figure 14 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is Total amount to be paid? | What is the value of the total amount to be paid? | BALANCE_DUE : 305.39 $ | [14](https://arxiv.org/html/2501.03403v3#A4.F14 "Figure 14 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is MANUFACTURER:? | What is the value of the manufacturer? | R. J. REYNOLDS | [15](https://arxiv.org/html/2501.03403v3#A4.F15 "Figure 15 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is BRAND NAME:? | What is the value of the brand name? | CARDINAL CIGARETTES (11 PACKINGS) | [15](https://arxiv.org/html/2501.03403v3#A4.F15 "Figure 15 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| What is OTHER INFORMATION:? | What is the value of OTHER INFORMATION? | SEE ATTACHMENT | [15](https://arxiv.org/html/2501.03403v3#A4.F15 "Figure 15 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| to whom is this letter written to? | to whom is this letter written to? | Mr. Rionda | [16](https://arxiv.org/html/2501.03403v3#A4.F16 "Figure 16 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| when is the letter dated ? | when is the letter dated ? | October 18, 1940, | [16](https://arxiv.org/html/2501.03403v3#A4.F16 "Figure 16 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| what is the auth. no. mentioned in the given form ? | what is the auth. no. mentioned in the given form ? | 5754 | [17](https://arxiv.org/html/2501.03403v3#A4.F17 "Figure 17 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| what is the value of percent per account as mentioned in the given form ? | what is the value of percent per account as mentioned in the given form ? | 50.06 | [17](https://arxiv.org/html/2501.03403v3#A4.F17 "Figure 17 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| what is the emp. no. mentioned in the given form ? | what is the emp. no. mentioned in the given form ? | 483378 | [17](https://arxiv.org/html/2501.03403v3#A4.F17 "Figure 17 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| what is the employee name mentioned in the given form ? | what is the employee name mentioned in the given form ? | IRENE KARL | [17](https://arxiv.org/html/2501.03403v3#A4.F17 "Figure 17 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| what is the value of amount authorized per account ? | what is the value of amount authorized per account ? | 292.00 | [17](https://arxiv.org/html/2501.03403v3#A4.F17 "Figure 17 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| Qual è Cognome? | Qual è Cognome? | ANNI | [18](https://arxiv.org/html/2501.03403v3#A4.F18 "Figure 18 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| Qual è Nome? | Qual è Nome? | GIACCOMO | [18](https://arxiv.org/html/2501.03403v3#A4.F18 "Figure 18 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| Qual è Data Nascita? | Qual è Data Nascita? | 12/02/1988 | [18](https://arxiv.org/html/2501.03403v3#A4.F18 "Figure 18 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| Qual è Data? | Qual è Data? | 19/12/2020 | [18](https://arxiv.org/html/2501.03403v3#A4.F18 "Figure 18 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |
| Qual è Ora? | Qual è Ora? | 14:00 | [18](https://arxiv.org/html/2501.03403v3#A4.F18 "Figure 18 ‣ Appendix D Dataset examples ‣ BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations") |

![Image 6: Refer to caption](https://arxiv.org/html/2501.03403v3/supplementary_img/app-deepform-sample.png)

Figure 8: Deepform sample

![Image 7: Refer to caption](https://arxiv.org/html/2501.03403v3/supplementary_img/app-charity-sample.png)

Figure 9: Kleister Charity sample

![Image 8: Refer to caption](https://arxiv.org/html/2501.03403v3/supplementary_img/app-nda-sample.png)

Figure 10: Kleister NDA sample

![Image 9: Refer to caption](https://arxiv.org/html/2501.03403v3/supplementary_img/app-ad-buy-sample.png)

Figure 11: VRDU Ad Buy Form. Not all the questions for this page are listed in Table LABEL:tab:app_samples, only until the details of the first broadcasting.

![Image 10: Refer to caption](https://arxiv.org/html/2501.03403v3/supplementary_img/app-registration-sample.png)

Figure 12: VRDU Registration Form sample.

![Image 11: Refer to caption](https://arxiv.org/html/2501.03403v3/supplementary_img/app-dude-sample.png)

Figure 13: DUDE sample.

![Image 12: Refer to caption](https://arxiv.org/html/2501.03403v3/supplementary_img/app-fatura-sample.png)

Figure 14: FATURA sample.

![Image 13: Refer to caption](https://arxiv.org/html/2501.03403v3/supplementary_img/app-funsd-sample.png)

Figure 15: FUNSD sample.

![Image 14: Refer to caption](https://arxiv.org/html/2501.03403v3/supplementary_img/app-mp-sample.png)

Figure 16: MP-DocVQA sample.

![Image 15: Refer to caption](https://arxiv.org/html/2501.03403v3/supplementary_img/app-sp-sample.png)

Figure 17: SP-DocVQA sample.

![Image 16: Refer to caption](https://arxiv.org/html/2501.03403v3/supplementary_img/app-xfund-sample.png)

Figure 18: XFUND sample.
