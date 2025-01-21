import dashscope
import json
from http import HTTPStatus

# Set your dashscope api_key
API_KEY = ""

dashscope.api_key = API_KEY

role = "user"

# news texts path
input_file = 'LIRA/train.json'

# rationale
output_response_file = 'train/rationale_train.json'

# request error
error_response_file = 'train/error_responses_train.json'

# json formate error
json_format_error_file = 'train/json_format_error_train.json'

rationale_response = []
error_response = []
json_format_error = []

def check_rationale_list(json_data):
    try:
        if len(json_data['rationale_list']) != 3:
            print("The number of rationale is not 3!")
            return False
        for rationale in json_data['rationale_list']:
            if not all(key in rationale for key in ["analytical_perspective", "rationale"]):
                print("The format of the key in the rationale list is incorrect!")
                return False
    except KeyError as e:
        return False
    return True

# read dataset and get the news texts
with open(input_file, 'r') as f:
    input_data = json.load(f)
content_values = [item['content'] for item in input_data]

returned_json_format = '''
"llm_predict": ,
"rationale_list": [
    {
        "analytical_perspective": "",
        "rationale": ""    
    },
    {
        "analytical_perspective": "",
        "rationale": ""
    },
    {
        "analytical_perspective": "",
        "rationale": ""
    }
]
'''

for content in content_values[701:]:
    user_message = [
        {'role': role,
         "content":
            f'''

            If the following text is  is fake news, please list: (1) provide a list of reasons from multiple analytical perspectives for each rationale, 
            and (2) the value of the 'llm_predict' field in the returned JSON is 1.
            If the following text is true news, please list: (1) the judgment criteria in the given text, 
            and (2) the value of the 'llm_predict' field in the returned JSON is 0.
            
            news content: {content}.
            
            Please note, only output in JSON format as follow, {returned_json_format}.
            
            Please note that it must be returned in the JSON format mentioned above.
            '''
         }
    ]
    response = dashscope.Generation.call(
        model=dashscope.Generation.Models.qwen_plus,
        messages=user_message,
        result_format='message'
    )
    if response.status_code == HTTPStatus.OK:
        r_content = response.output.choices[0].message.content
        print(f"rationale list =\n{r_content}")
        print(f"news text = {content}")

        if r_content.startswith("```json") and r_content.endswith("```"):
            r_content_cleaned = r_content.removeprefix("```json").removesuffix("```")
            try:
                r_content_json = json.loads(r_content_cleaned)
                r_content_json['content'] = content
                # check JSON format
                if check_rationale_list(r_content_json):
                    rationale_response.append(r_content_json)
                    print(f"rationale list length: , {len(rationale_response)}")
                    with open(output_response_file, 'w', encoding='utf-8') as f:
                        json.dump(rationale_response, f, ensure_ascii=False, indent=4)
                else:
                    json_format_error.append(content)
            except json.decoder.JSONDecodeError as e:
                json_format_error.append(content)
        else:
            json_format_error.append(content)
        with open(json_format_error_file, "w", encoding='utf-8') as f:
            json.dump(json_format_error, f, ensure_ascii=False, indent=4)
    else:
        error_info = {
            "request_id": response.request_id,
            'status_code': response.status_code,
            'error_code': response.code,
            'error_message': response.message,
            'content_news': content
        }
        error_response.append({"error": error_info})
        with open(error_response_file, 'w') as f:
            json.dump(error_response, f, ensure_ascii=False, indent=4)
