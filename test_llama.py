# import ollama
import time
from llama_index.llms.ollama import Ollama
import random
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


llm = Ollama(model="llama3", request_timeout=60.0)

def prompt_llama(prompt):
    start = time.time()
    response = llm.complete(prompt)#'what is 2+2')
    end = time.time()
    runtime = end-start
    return response, runtime


def generate_number(lowbound = 0, upbound = 1, numgenerate = 10, numdigits = 2):
    return [round(random.uniform(lowbound, upbound), numdigits) for _ in range(numgenerate)]

def generate_sequence(numdigits = 2, maxterms = 64):
    # return Xs, Ys
    weight = generate_number(numgenerate = 1, numdigits = numdigits)[0]
    Xs = generate_number(numgenerate=maxterms, numdigits = numdigits)
    Ys = [weight*x for x in Xs]
    return Xs, Ys, weight

def format_number(num, decimalplaces):
    totaldigits = decimalplaces+2
    roundnum = round(num, decimalplaces)
    num_str = str(roundnum)
    num_str = num_str[:totaldigits]
    if not len(num_str)==totaldigits:#+2 for the 0.
        num_str = num_str + '0'*(totaldigits-len(num_str))
    return num_str
    

def generate_prompt1(numdigits = 2, maxterms = 8):
    # give it just a sequence of numbers, don't let the model know it's a regression task
    Xs, Ys, weight= generate_sequence(numdigits = numdigits, maxterms = maxterms+1) # maxtersm +1 bc last number we use for answer
    prompt = f"Give me the next number in the sequence. Provide one number with {numdigits*2} decimal places. Do not respond with anything except the number.\n\n"
    answer = Ys[-1]
    for i in range(len(Xs)-1):
        xstr, ystr = format_number(Xs[i], numdigits), format_number(Ys[i], 2*numdigits)
        prompt += f"{xstr}, {ystr}, "
    prompt+=f"{format_number(Xs[-1], numdigits)}, "
    return prompt, answer

def generate_prompt2(numdigits = 2, maxterms = 8):
    #arxiv.org/pdf/2404.07544.pdf method taken from here, appendix C.2
    Xs, Ys, weight = generate_sequence(numdigits = numdigits, maxterms = maxterms+1)
    prompt = 'The task is to provide your best estimate for "Output". Please provide that and only that, without any additional text.\n\n'
    answer = Ys[-1] 
    for i in range(len(Xs)-1):
        xstr, ystr = format_number(Xs[i], numdigits), format_number(Ys[i], 2*numdigits)
        prompt += f"Feature 0: {xstr}\nOutput: {ystr}\n\n"
    prompt += f"Feature 0: {format_number(Xs[-1], numdigits)}\nOutput: "
    return prompt, answer

def generate_prompt(promptstyle = 2, numdigits = 2, maxterms = 8):
    if promptstyle == 1:
        return generate_prompt1(numdigits = numdigits, maxterms = maxterms)
    elif promptstyle == 2:
        return generate_prompt2(numdigits = numdigits, maxterms = maxterms)


def test_llama_ICL(maxtermstotest = 14, runeach = 5, numdigits = 2, promptstyle = 2):
    results = {'prompt': [], 'answer':[], 'response': [], 'numdigits': [], 'maxterms': [], 'runnumber': [], 'runtime': []}
    for i in range(1, maxtermstotest+1):
        for j in range(runeach):
            print(f"Testing {i} terms, run {j+1}/{runeach}")
            prompt, answer = generate_prompt(promptstyle = promptstyle, numdigits = numdigits, maxterms = i)
            print(prompt)
            ahat, runtime = prompt_llama(prompt)
            print(f"Answer: {answer}, Response: {ahat}| Time: {runtime:.2f}s")
            results['prompt'].append(prompt)
            results['answer'].append(answer)
            results['response'].append(ahat)
            results['numdigits'].append(numdigits)
            results['maxterms'].append(i)
            results['runnumber'].append(j)
            results['runtime'].append(runtime)
        df = pd.DataFrame(results)
        df.to_csv(f"dfs/llama_ICL_{numdigits}digits_pstyle{promptstyle}.csv", index = False)

def extract_floats(input_string):
    numbers = re.findall(r'\d+\.\d+|\d+', input_string)
    return numbers[0]

def make_num(series):
    vals = []
    for i in range(len(series)):
        val = series[i]
        val = extract_floats(str(val))
        vals.append(float(val))
    return np.array(vals)

def plot_ICL(numdigits = 2, promptstyle = 2):
    df = pd.read_csv(f"dfs/llama_ICL_{numdigits}digits_pstyle{promptstyle}.csv")
    cl = []
    mses = []
    mse_stds = []
    for maxterm in df['maxterms'].unique():
        dfterm = df[df['maxterms']==maxterm]
        responses = make_num(np.array(dfterm['response']))
        answers = make_num(np.array(dfterm['answer']))
        mse_resp = (responses-answers)**2
        cl.append(maxterm)
        mses.append(mse_resp.mean())
        mse_stds.append(mse_resp.std())
    plt.plot(cl, mses,marker = 'o', linestyle = '-', color = 'b')
    # plot yerr
    #plt.errorbar(cl, mses, yerr = mse_stds, fmt = 'o', color = 'b')

    plt.xlabel("Number of Examples Given to LLama3 In Context")
    plt.ylabel("MSE")
    #plt.yscale('log')
    plt.title('MSE of LLama3 on ICL Linear Regression task')
    plt.show()



 

if __name__ == "__main__":
    plot_ICL()
    #generate_sequence()
    #test_llama_ICL(maxtermstotest = 16, runeach = 5, numdigits = 2, promptstyle = 2)
