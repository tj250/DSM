import langextract as lx
import textwrap
import os
from collections import Counter, defaultdict
import markdown

# Define comprehensive prompt and examples for complex literary text
prompt = textwrap.dedent("""\
    Extract environmental covariates, soil properties, and the relationships between environmental covariates and soil properties from the given text.

Provide meaningful attributes for each entity to add context and depth.

Important Note 1: When extracting text, please use the exact wording from the input text; do not rephrase.
Extract entities in the order they appear, and the text spans must not overlap.""")

examples = [
    lx.data.ExampleData(
        text=textwrap.dedent("""\
            摘要: 【目的】开展黄土高原不同地貌区农田土壤有机质(SOM)预测方法研究，探讨不同预测方法在不同区域的适用性及不确定性，以便更准确地估算农田SOM空间分布特征，对土壤资源高效利用和农田精细化管理具有重要意义。
            【方法】在黄土高原3种典型地貌区进行试验，包括丘陵沟壑区(庄浪县)、高塬区(宁县)和平原区(武功县)，分别布设样点 3788、4048 和 3860 个，分析农田土壤 0—20cmSOM含量。
            运用地统计学理论，分析各典型区SOM空间分布特征。提取原始样本75%为建模点，其余25%为验证点，利用普通克里格(OK)、随机森林(RF)和随机森林+普通克里格(RF+OK)等方法，
            结合土壤类型、地形、气候、植被、人类活动等多源影响因子，对SOM分布进行空间预测，并对预测结果进行误差分析和空间结构检验，明确各方法的不确定性。
            【结果】丘陵沟壑区、高塬区、平原区SOM平均含量分别为14.29、13.15、14.48g/kg，均属于较低水平；变异系数分别为 18.96%、19.54%、26.71%，呈中等变异；
            块金效应分别为 8.60%、17.41% 和10.01%，受随机性和结构性因子共同作用，且受后者影响更大；丘陵沟壑区和平原区 SOM 含量的 Moran’s I 分别为0.26和0.14，Z[I] 分别为 26.56 和 13.51，
            存在显著空间自相关性，而高塬区SOM含量 Moran’s I为 0.02，Z[I]为1.55，不存在空间自相关性。
            丘陵沟壑区、高塬区、平原区SOM含量空间分布分别受温度、海拔、降水影响最大。在平原区，RF+OK 法较 RF 法和 OK 法，MSE、RMSE、MAE 等误差均最小，
            实测值与预测值的相关系数(r)最高，预测值的空间结构与实测值更接近。高塬区SOM空间分布无规律，OK法在该区域不适用，RF法和RF+OK法的各项误差无明显差异，但RF法的r更高，
            且预测值的空间结构更符合宁县实际特征。在平原区，OK法预测结果的不确定性较大，RF和RF+OK 方法各项误差和r均无明显差异，但RF方法预测值的空间结构与实测值更接近，
            且较其它两个地区，其SOM变异性及建模点和验证点的各项误差均最大。
            【结论】在不同地貌区，环境要素、空间结构不同，同一预测方法的预测精度存在差异，平原区较丘陵沟壑区和高塬区，其空间预测结果的不确定性更大。在同一地貌区，3种预测方法的预测结果存在差异，
            丘陵沟壑区使用RF+OK法预测SOM空间分布效果较好，而高塬区和平原区则用 RF 法较好。当区域SOM存在显著空间相关性，且半方差函数的拟合度较高、残差较小时，采用RF+OK 方法可显著提高模型预测精度。"""),
        extractions=[
            lx.data.Extraction(
                extraction_class="土壤属性",
                extraction_text="SOM",
                attributes={"研究区域": "黄土高原的混合地貌"}
            ),
            lx.data.Extraction(
                extraction_class="环境协变量",
                extraction_text="温度、海拔、降水",
                # attributes={"类型": "地形、气候"}
            ),
            # lx.data.Extraction(
            #     extraction_class="环境协变量与土壤属性的关系",
            #     extraction_text="海拔影响土壤SOM",
            #     attributes={"type": "影响", "环境协变量": "海拔", "土壤属性": "SOM"}
            # ),
            # lx.data.Extraction(
            #     extraction_class="环境协变量与土壤属性的关系",
            #     extraction_text="气候影响土壤SOM",
            #     attributes={"type": "影响", "环境协变量": "气候", "土壤属性": "SOM"}
            # ),
            # lx.data.Extraction(
            #     extraction_class="环境协变量与土壤属性的关系",
            #     extraction_text="降水影响土壤SOM",
            #     attributes={"type": "影响", "环境协变量": "降水", "土壤属性": "SOM"}
            # ),
        ]
    ),
    # lx.data.ExampleData(
    #     text=textwrap.dedent("""\
    #     Wildfires shape the biogeochemistry of a landscape, and burn history is prevalent in US Southwestern mixed-conifer forests (Swetnam and Betancourt, 1983).
    #     Novel, high-severity wildfires can have lasting, decadal impacts on regional soil biogeochemistry in mixed-conifer forests (Dove et al. 2020).
    #     Differences in fire severity often create a mosaic landscape with variations in vegetation survival, substrate type, and C and N availability that influence microbial community structure and function (Barros et al., 2012).
    #     Wildfire is shaped by topography, tending to move upslope leading to higher-severity burn in elevated areas (Barros et al., 2012; Debano, 2000; Linn et al., 2007).
    #     This can amplify topographic patterns of microbial activities by exacerbating soil biogeochemical differences in convergent versus planar areas (Dillon et al., 2011).
    #     Fire also shows differential vertical effects throughout the soil profile through removal of surface organic matter, deterioration of soil structure and porosity, loss of carbon, nitrogen and other nutrients through volatilization, leaching, erosion, and marked alteration of microbial communities that result from direct (surface-burning) and indirect fire impacts (alterations to soil physicochemical environment and vegetation cover).
    #     (Bento-Gonçalves et al., 2012; Ferrenberg et al., 2013; Jiménez Esquilín et al., 2007; Weber et al., 2014).
    #     These soil fire effects are often most significant in the 0-5 cm layer that generally contains high OM content, but the impacts of fire extend into the soil profile in a manner dependent on the site and fire characteristics, i.e. soil type, topographic position, vegetation cover.
    #      (Dooley and Treseder, 2012; Holden and Treseder, 2013; Jimenez Esquilín et al., 2007).
    #      Further, the magnitude of fire effects on soil properties correlate directly with burn severity (Certini, 2005; Knelman et al., 2015; Murphy et al., 2006, Lybrand et al., 2018)"""),
    #     extractions=[
    #         lx.data.Extraction(
    #             extraction_class="soil properties",
    #             extraction_text="SOM",
    #             attributes={"Geographic area": "黄土高原不同地貌区"}
    #         ),
    #         lx.data.Extraction(
    #             extraction_class="environmental covariates",
    #             extraction_text="temperature, LAI, rainfall, humidity, forest",
    #             attributes={"covariate type": "terrain"}
    #         ),
    #         lx.data.Extraction(
    #             extraction_class="relationship",
    #             extraction_text="Juliet is the sun",
    #             attributes={"type": "metaphor", "character_1": "Romeo", "character_2": "Juliet"}
    #         ),
    #     ]
    # )
]

# Process Romeo & Juliet directly from Project Gutenberg
print("遍历待处理的Markdown文本...")
current_directory = os.getcwd()
for root, dirs, files in os.walk(current_directory):
    for file in files:
        file_path = os.path.join(root, file)
        if not file_path.endswith('.md'):
            continue
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
            # Process Romeo & Juliet directly from Project Gutenberg
            plain_text = markdown.markdown(markdown_text, extensions=['markdown.extensions.extra'])
            result = lx.extract(
                text_or_documents=plain_text,
                prompt_description=prompt,
                examples=examples,
                model_id="gemma3:27b-it-qat",  # or any Ollama model
                model_url="http://192.168.1.71:11434",
                fence_output=False,
                use_schema_constraints=False,
                extraction_passes=3,      # Multiple passes for improved recall
                max_workers=10,           # Parallel processing for speed
                max_char_buffer=1000      # Smaller contexts for better accuracy
            )

            print(f"Extracted {len(result.extractions)} entities from {len(result.text):,} characters")

            # Save and visualize the results
            lx.io.save_annotated_documents([result], output_name=file_path + ".jsonl", output_dir=".")

            # Generate the interactive visualization
            html_content = lx.visualize(file_path + ".jsonl")
            with open(file_path + "_visualization.html", "w", encoding="utf-8") as f:
                if hasattr(html_content, 'data'):
                    f.write(html_content.data)  # For Jupyter/Colab
                else:
                    f.write(html_content)

            # print("Interactive visualization saved to romeo_juliet_visualization.html")


            # Analyze character mentions
            characters = {}
            for e in result.extractions:
                if e.extraction_class == "character":
                    char_name = e.extraction_text
                    if char_name not in characters:
                        characters[char_name] = {"count": 0, "attributes": set()}
                    characters[char_name]["count"] += 1
                    if e.attributes:
                        for attr_key, attr_val in e.attributes.items():
                            characters[char_name]["attributes"].add(f"{attr_key}: {attr_val}")

            # Print character summary
            print(f"\nCHARACTER SUMMARY ({len(characters)} unique characters)")
            print("=" * 60)

            sorted_chars = sorted(characters.items(), key=lambda x: x[1]["count"], reverse=True)
            for char_name, char_data in sorted_chars[:10]:  # Top 10 characters
                attrs_preview = list(char_data["attributes"])[:3]
                attrs_str = f" ({', '.join(attrs_preview)})" if attrs_preview else ""
                print(f"{char_name}: {char_data['count']} mentions{attrs_str}")

            # Entity type breakdown
            entity_counts = Counter(e.extraction_class for e in result.extractions)
            print(f"\nENTITY TYPE BREAKDOWN")
            print("=" * 60)
            for entity_type, count in entity_counts.most_common():
                percentage = (count / len(result.extractions)) * 100
                print(f"{entity_type}: {count} ({percentage:.1f}%)")