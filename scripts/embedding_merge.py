# stable-diffusion-webui-embedding-merge

import re
import os
import torch
import json
import time
import html
import traceback
import threading
import gradio
import modules.extras
import modules.ui
from modules.shared import opts, cmd_opts
from modules import shared, scripts, script_callbacks, processing, devices, styles
from modules.processing import StableDiffusionProcessing
from webui import wrap_gradio_gpu_call
from modules.textual_inversion.textual_inversion import Embedding

from ldm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder
import open_clip.tokenizer

_webui_embedding_merge_extension_ = None

class EmbeddingMergeExtension(scripts.Script):
    def title(self):
        return 'Embedding Merge'
    def show(self,is_img2img):
        return scripts.AlwaysVisible
    def process(self,p):
        if _webui_embedding_merge_extension_ is not None:
            _webui_embedding_merge_extension_(p)

class Exception_From_EmbeddingMergeExtension(Exception):
    pass
class Exception_From_EmbeddingMergeExtension_():
    def __init__(self,_):
        self._ = _
    def __getattr__(self,_):
        raise Exception_From_EmbeddingMergeExtension(self._)

def _webui_embedding_merge_():

    def gr_tab():
        with gradio.Blocks(analytics_enabled=False) as block:
            gradio.HTML('<style>#tab_embedding_merge_extension p::before,#tab_embedding_merge_extension p::after,#tab_embedding_merge_extension code::before,#tab_embedding_merge_extension code::after{display:none!important}</style>')
            with gradio.Row():
                with gradio.Accordion('Embedding Merge extension! (Click here for usage instructions)', open=False):
                    gradio.Markdown("## View text or embeddings vectors\n\nYou can paste your vanilla prompt (without any other special syntax) into the textbox below to see how it is parsed by WebUI. All of detected Textual Inversion embeddings will be extracted and presented to you along with literal text tokens. For example:\n\n>intergalactic train, masterpiece, by Danh Víµ")
                    with gradio.Accordion('More about table columns and grouping of its rows...', open=False):
                        gradio.Markdown('### Rows:\n\n- `By none` = interpret the prompt as a whole, extracting all characters from real tokens\n- `By comma` = split the prompt by tags on commas, removing commas but keeping source space characters\n- `By parts` (default) = split at TI embeddings, joining text parts together, keeping spaces\n- `By words` = split only after tokens that actually produce space character at the end\n- `By tokens` = split at everything except characters that are represented with more than one vector\n- `By vectors` = show all vectors separated, even for TI embeddings\n\n### Columns:\n\n- `Index` = index of one vector or index range (inclusive) for this row\n- `Vectors` = number of final vectors for this row (to clearly see it)\n- `Text` = original or recreated from tokens text, enclosed in quotes for clarity\n- `Token` = list of CLIP token numbers that represent this row; for TI embeddings \\* or \\*_X where X is the index of current embedding vector\n- `Min` = lowest (negative) value of the vector or grouped vectors values\n- `Max` = largest value\n- `Sum` = sum of all values with sign\n- `Abs` = sum of modulus of each value, without sign (always positive)\n- `Len` = vector length in L2 norm, square root of sum of squared values (computed approximate)')
                    gradio.Markdown("## Test merge expression:\n\nYou can enter a \"merge expression\" that starts with a single quote, to see how it will be parsed and combined by this extension. It should contain single quotes around literal texts or TI embeggings, and special operators between them. For example:\n\n>'greg rutkowski'/4+'gustav dore'*0.75")
                    with gradio.Accordion('More about merge expression syntax...', open=False):
                        gradio.Markdown("- `  'one' + 'two'  ` = blend vectors together by simple sum of all values. If length is different, smallest part will be right-padded with zeroes.\n\n- `  'one' - 'two'  ` = as above, but subtraction. Note that + and - can be put only between textual parts and will have lowest priority.\n\n- `  'text' * NUM  ` = multiply all vectors of quoted literal by numeric value. You can use floating point (0.85) and negative numbers (-1), but not arithmetic expressions.\n\n- `  'text' / NUM  ` = division by number, just as multiplication above. Applies to previous text literal but after similar operations, so you can multiply and divide together (\*3/5)\n\n- `  'text' : NUM  ` = change vector count of literal, to shrink or enlarge (padded with zeros). Only integer without sign!\n\n- `  'text' :+ NUM  ` and `  'text'  :- NUM  ` = circular rotate vectors in this token, for example +1 will shift index of each vector by one forward, wrapping on last.\n\nTo apply multiplication (or division), cropping or shifting *to the result* of addition (or subtraction), you cannot use parenthesis; instead, try this syntax:\n\n- `  'one' + 'two' =* NUM ` = will multiply the sum of 'one' and 'two', but not 'two' alone\n\n- `  'one' + 'two' =/ NUM  ` = divide the sum (or any number of sums to the left), effectively the \"result\" of everything\n\n- `  'one' + 'two' =: NUM  ` = crop or enlarge the results\n\n- `  'one' + 'two' =:+ NUM  ` or `  'one' + 'two' =:- NUM  ` = rotate the result\n\nThus, the following operations are doing the same:\n\n>`  'a'/2 + 'b'/2 + '':1 - 'd'  `   \n`  'a'+'b' =* 0.5 + 'c'*0 + 'd'*-1  `")
                    gradio.Markdown("## Several merge expressions in prompt:\n\nIf you put a valid merge expression enclosed in angular <'…' …> or curly {'…' …} brackets anywhere in your prompt (with no space between `<` or `{` and `'`), it will be parsed and merged into one temporary Textual Inversion embedding, which replaces the expression itself. The resulting prompt will be joined from those embeddings and anything between expressions. For example:\n\n>A photo of <'cat'+'dog'>, {'4k'+'dynamic lighting'+'science fiction'=/3} masterpiece")
                    with gradio.Accordion('More examples of using angular/curly brackets...', open=True):
                        gradio.Markdown('TODO')
                    gradio.Markdown("## Using merge expressions in prompts at runtime!\n\nYou can actually put merge expressions in angular or curly brackets into your txt2img or img2img prompt in WebUI. This extension will intercept both main and negative prompts, parse and merge expressions creating temporary TI embeddings that WebUI will \"see\" instead of your original text. In generation info there will be internal meaningless names like <'EM_1'>, but extra parameter \"EmbeddingMerge\" will contain original merge expressions. To quickly restore your prompts, just paste your complete generation information (from .txt or PNG Info) into the textbox on this tab (also it should work for the official \"paste\" toolbar button too) – its temporary embeddings will be replaced back with expressions, for example:\n\n> a photo of <'EM_1'>  \nNegative prompt: {'EM_2'}  \nSteps: 8, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1374372309, Size: 512x512, Model hash: c6bbc15e32, Model: sd-v1-5-inpainting, EmbeddingMerge: \"<'EM_1'>=<'sky' * 2/4 + 'forest' * 3/4>, {'EM_2'}={'blurry'+'cropped'}\", Conditional mask weight: 1");
                    with gradio.Accordion('Technical information...', open=True):
                        gradio.Markdown('TODO')
            with gradio.Row():
                gr_text = gradio.Textbox(value='', lines=4, max_lines=16, interactive=True, label='Your prompt (no weight/attention, do not escape parenthesis/brackets); or your merge expression (if the first character is a single quote); or a generation info to restore prompts')
            with gradio.Row():
                with gradio.Column(scale=1):
                    gr_button = gradio.Button('Parse!',variant='primary')
                with gradio.Column(scale=3):
                    gr_radio = gradio.Radio(choices=('By none','By comma','By parts','By words','By tokens','By vectors'), value='By parts', type='index', interactive=True, label='Group/split table by: (when not started with single quote - so only for prompts, not for merge)')
            with gradio.Box():
                gr_html = gradio.HTML(label='out')
            with gradio.Row():
                gr_true = gradio.Checkbox(value=True,visible=False,show_label=False)
                gr_false = gradio.Checkbox(value=False,visible=False,show_label=False)
                gr_name = gradio.Textbox(value='', lines=1, max_lines=1, interactive=True, label='Type here a name for your new embedding that will store the result of next parsing/merging by the button above: (optional; cleared on success)')
            gr_button.click(fn=gr_func, inputs=[gr_name,gr_text,gr_radio,gr_true], outputs=[gr_html,gr_name,gr_text], show_progress=False)
            gr_radio.change(fn=gr_func, inputs=[gr_name,gr_text,gr_radio,gr_false], outputs=[gr_html,gr_name,gr_text], show_progress=False)
        return [(block,'EM','embedding_merge_extension')]

    def tokens_to_text():
        try:
            # https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer
            class VanillaClip:
                def __init__(self, clip):
                    self.clip = clip
                def vocab(self):
                    return self.clip.tokenizer.get_vocab()
                def byte_decoder(self):
                    return self.clip.tokenizer.byte_decoder
            class OpenClip:
                def __init__(self, clip):
                    self.clip = clip
                    self.tokenizer = open_clip.tokenizer._tokenizer
                def vocab(self):
                    return self.tokenizer.encoder
                def byte_decoder(self):
                    return self.tokenizer.byte_decoder
            clip = shared.sd_model.cond_stage_model.wrapped
            if isinstance(clip, FrozenCLIPEmbedder):
                clip = VanillaClip(shared.sd_model.cond_stage_model.wrapped)
            elif isinstance(clip, FrozenOpenCLIPEmbedder):
                clip = OpenClip(shared.sd_model.cond_stage_model.wrapped)
            else:
                return None
            vocab = {v: k for k, v in clip.vocab().items()}
            byte_decoder = clip.byte_decoder()
            def _tokens_to_text(tokens):
                nonlocal vocab, byte_decoder
                code = []
                ids = []
                current_ids = []
                class_index = 0
                def dump(last=False):
                    nonlocal code, ids, current_ids
                    words = [vocab.get(x, '') for x in current_ids]
                    try:
                        word = bytearray([byte_decoder[x] for x in ''.join(words)]).decode('utf-8')
                    except UnicodeDecodeError:
                        if last:
                            word = '<ERR>' * len(current_ids)
                        elif len(current_ids) > 4:
                            id = current_ids[0]
                            ids += [id]
                            local_ids = current_ids[1:]
                            code += [([id], '<ERR>')]

                            current_ids = []
                            for id in local_ids:
                                current_ids.append(id)
                                dump()
                            return
                        else:
                            return
                    word = word.replace('</w>', ' ')
                    code += [(current_ids, word)]
                    ids += current_ids
                    current_ids = []
                for token in tokens:
                    token = int(token)
                    current_ids.append(token)
                    dump()
                dump(last=True)
                return [c for c in code if len(c[0])!=0]
            return _tokens_to_text
        except:
            traceback.print_exc()
            return None

    def str_to_escape(line):
        res = re.sub(r'([()[\]\\])',r'\\\1',line)
        return res

    def text_to_vectors(text):
        dv = None
        dt = None
        try:
            res = []
            text = text.lstrip().lower()
            clip = shared.sd_model.cond_stage_model
            tokens = clip.tokenize_line(str_to_escape(text))
            count = tokens[1]
            tokens = tokens[0][0]
            fixes = tokens.fixes
            if count>=len(tokens.tokens):
                return None
            tokens = tokens.tokens[1:count+1]
            start = 0
            for fix in fixes:
                name = fix.embedding.name.lower()
                tensor = fix.embedding.vec
                num = fix.embedding.vectors
                off = fix.offset
                if num!=tensor.size(0):
                    return None
                lenname = len(name)
                if off!=start:
                    test = 0
                    while True:
                        pos = text.find(name,test)
                        if pos<0:
                            return None
                        test = pos+lenname
                        sub = text[0:test]
                        part = clip.tokenize_line(str_to_escape(sub))
                        cnt = part[1]
                        part = part[0][0]
                        vec = off-start
                        need = tokens[start:off+num]
                        if part.tokens[1:cnt+1]==need:
                            trans = clip.encode_embedding_init_text(text,vec)
                            t = trans[:vec].to(device=devices.device,dtype=torch.float32)
                            res.append((t,sub[:pos],need[:vec]))
                            text = text[pos:]
                            start = off
                            break
                if text[0:lenname]!=name:
                    return None
                tensor = tensor.to(device=devices.device,dtype=torch.float32)
                res.append((tensor,name,None))
                start += num
                text = text[lenname:].lstrip()
            if text!='':
                part = clip.tokenize_line(str_to_escape(text))
                cnt = part[1]
                part = part[0][0]
                need = tokens[start:]
                if part.tokens[1:cnt+1]!=need:
                    return None
                trans = clip.encode_embedding_init_text(text,999)
                trans = trans.to(device=devices.device,dtype=torch.float32)
                res.append((trans,text,need))
            return res
        except:
            traceback.print_exc()
            return None

    def text_to_tokens(text):
        try:
            tokens = shared.sd_model.cond_stage_model.tokenize([text])[0]
            return tokens
        except:
            return None

    def to_float(num):
        if num is None: 
            return None
        try:
            return float(num)
        except:
            return None

    def grab_vectors(text):
        try:
            res = text_to_vectors(text)
            if res is None:
                return None
            if len(res)==0:
                res = text_to_vectors(',')[0][0][0:0]
                return res
            res = torch.cat([ten[0] for ten in res]);
            return res
        except:
            return None

    reg_clean = re.compile(r'\s+')
    reg_oper = re.compile(r'(=?)(?:([*/])([+-]?[0-9]*(?:\.[0-9]*)?)|:([+-]?)(-?[0-9]+))')

    def gr_parser(text):
        orig = '"'+text+'"'
        text = text.replace('\0',' ')+' '
        length = len(text)
        arr = []
        left = 0
        quot = False
        join = False
        while left<length:
            pos = text.find("'",left)
            if pos<0:
                pos = length
            take = text[left:pos]
            if left>0:
                if take=='' and not quot:
                    join = True
                elif quot:
                    if join:
                        arr[-1] = (arr[-1][0]+"'"+take,True)
                        join = False
                    else:
                        arr.append((take,True))
                else:
                    arr.append((take.strip(),False))
            quot = not quot
            left = pos+1
        if not quot:
            return (None,'Last quote not closed in '+orig)
        if len(arr)>0 and arr[-1][0]=='':
            arr.pop()
        
        actions = []
        for param, quot in arr:
            one = param
            if quot:
                actions.append({
                  'A': None,
                  'V': param,
                  'O': one,
                })
                continue
            param = reg_clean.sub('',param)
            while param!='':
                m = reg_oper.match(param)
                if not m:
                    if param=='+' or param=='-':
                        actions.append({
                          'A': False,
                          'V': param=='+',
                          'O': one,
                        })
                        break
                    return (None,'Wrong expression "'+param+'" in '+orig)
                m_flag = m.group(1)=='='
                m_mul = m.group(2)
                m_val = m.group(3)
                m_shift = m.group(4)
                m_size = m.group(5)
                if m_val is not None:
                    m_val = to_float(m_val)
                    if m_val is None:
                        return (None,'Bad param for multiplication "'+param+'" in '+orig)
                    m_mul = m_mul=='*'
                    m_size = -1
                    m_shift = 0
                else:
                    m_size = int(m_size)
                    if m_shift=='+':
                        m_shift = m_size
                        m_size = -1
                    elif m_shift=='-':
                        m_shift = -m_size
                        m_size = -1
                    else:
                        m_shift = 0
                    m_val = 1
                    m_mul = None
                actions.append({
                  'A': True,
                  'V': m_val,
                  'W': m_mul,
                  'S': m_size,
                  'R': m_shift,
                  'F': m_flag,
                  'O': one,
                })
                param = param[len(m.group(0)):]
        actions.append({
          'A': None,
          'V': None,
        })
        can_file = True
        can_add = False
        can_mul = False
        for act in actions:
            act['M'] = False
            A = act['A']
            if A==None:
                if act['V']==None:
                    if can_file:
                        return (None,'Need quoted string after last + or - in '+orig)
                    act['M'] = True
                    break
                if can_file:
                    can_add = True
                    can_mul = True
                    can_file = False
                else:
                    return (None,'Quoted string without preceding + or - at \''+act['O']+'\' in '+orig)
            elif A==True:
                if can_mul:
                    can_file = False
                    can_add = True
                    can_mul = True
                    if act['F']:
                        act['M'] = True
                else:
                    return (None,'Cannot multiply or modify at "'+act['O']+'" in '+orig)
            else:
                if can_add:
                    can_file = True
                    can_mul = False
                    can_add = False
                    act['M'] = True
                else:
                    return (None,'Cannot merge at "'+act['O']+'" in '+orig)
        left = None
        right = None
        add = 0
        for act in actions:
            if act['M'] and (left is not None):
                if add!=0:
                    (vectors1,length1) = left.size()
                    (vectors2,length2) = right.size()
                    if length1!=length2:
                        return (None,'Cannot merge different embeddings in '+orig)
                    if vectors1!=vectors2:
                        if vectors1<vectors2:
                            target = torch.zeros(vectors2,length1).to(device=devices.device,dtype=torch.float32)
                            target[0:vectors1] = left
                            left = target
                        else:
                            target = torch.zeros(vectors1,length2).to(device=devices.device,dtype=torch.float32)
                            target[0:vectors2] = right
                            right = target
                    if add>0:
                        right = left+right
                    else:
                        right = left-right
                left = None
            A = act['A']
            if A==None:
                line = act['V']
                if line==None:
                    return (right,None)
                right = grab_vectors(line)
                if right==None:
                    return (None,'Failed to parse \''+line+'\' in '+orig)
            elif A==False:
                if act['V']:
                    add = 1
                else:
                    add = -1
                left = right
                right = None
            else:
                s = act['S']
                r = act['R']
                if r!=0:
                    right = right.roll(r,dims=0)
                else:
                    if s>=0:
                        (vectors,length) = right.size()
                        if vectors>s:
                            right = right[0:s]
                        elif vectors<s:
                            target = torch.zeros(s,length).to(device=devices.device,dtype=torch.float32)
                            target[0:vectors] = right
                            right = target
                    elif act['W']==True:
                        right = right*act['V']
                    elif  act['W']==False:
                        right = right/act['V']
        return (right,None)

    def grab_embedding_cache():
        db = modules.sd_hijack.model_hijack.embedding_db
        field = '__embedding_merge_cache'
        if hasattr(db,field):
            cache = getattr(db,field)
        else:
            cache = {'_':0,'-':0}
            setattr(db,field,cache)
        return cache
        
    def register_embedding(name,embedding):
        # /modules/textual_inversion/textual_inversion.py
        self = modules.sd_hijack.model_hijack.embedding_db
        model = shared.sd_model
        try:
            ids = model.cond_stage_model.tokenize([name])[0]
            first_id = ids[0]
        except:
            return
        if embedding is None:
            if self.word_embeddings[name] is None:
                return
            del self.word_embeddings[name]
        else:
            self.word_embeddings[name] = embedding
        if first_id not in self.ids_lookup:
            if embedding is None:
                return
            self.ids_lookup[first_id] = []
        save = [(ids, embedding)] if embedding is not None else []
        old = [x for x in self.ids_lookup[first_id] if x[1].name!=name]
        self.ids_lookup[first_id] = sorted(old + save, key=lambda x: len(x[0]), reverse=True)
        return embedding

    def make_temp_embedding(name,vectors,cache):
        if (name is None) or (name==''):
            return
        name = name.strip()
        shape = vectors.size()
        if name in cache:
            embed = cache[name]
        else:
            embed = Embedding(vectors,name)
            cache[name] = embed
        embed.vec = vectors
        embed.step = None
        embed.vectors = shape[0]
        embed.shape = shape[-1]
        embed.filename = ''
        register_embedding(name,embed)
    
    def reset_temp_embeddings(prod):
        cache = grab_embedding_cache()
        prod = '_' if prod else '-'
        num = cache[prod]
        cache[prod] = 0
        for a,b in (('<','>'),('{','}')):
            i = num
            while i>0:
                tgt = a+"'EM"+prod+str(i)+"'"+b
                if tgt in cache:
                    embed = cache[tgt]
                    embed.vec = None
                    embed.shape = None
                    embed.vectors = 0
                    embed.cached_checksum = None
                i = i-1
        return cache

    def add_temp_embedding(vectors,cache,prod,curly):
        prod = '_' if prod else '-'
        num = 1+(cache[prod] or 0)
        name = "'EM"+prod+str(num)+"'"
        if curly:
            name = '{'+name+'}'
        else:
            name = '<'+name+'>'
        cache[prod] = num
        if name in cache:
            embed = cache[name]
            embed.vec = vectors
            shape = vectors.size()
            embed.vectors = shape[0]
            embed.shape = shape[-1]
            embed.cached_checksum = None
        make_temp_embedding(name,vectors,cache)
        return name
    
    def parse_infotext(text):
        orig = text
        pos = re.search(r"\bEmbeddingMerge:\s*(\"?[<{])'EM_",text)
        if pos is None:
            return (None,orig)
        head = text[:pos.span(0)[0]].rstrip()
        if len(head)>0 and head[-1]==',':
            head = head[:-1]
        text = text[pos.span(1)[0]:]
        if len(text)<2:
            return (None,orig)
        what = text[0]
        if what=='"':
            unquoted = None
        else:
            if what=='<':
                unquoted = '>'
            elif what=='{':
                unquoted = '}'
            else:
                return (None,orig)
        if unquoted is not None:
            stop = min_or_all(text.find(unquoted+','),text.find(unquoted+'\n'),-1)
            if stop<0:
                return (None,orig)
            stop += 1
            tail = text[stop:]
            line = text[:stop]
        else:
            stop = (text+'\n').find('\n')
            part = text[:stop]
            left = 0
            while True:
                right = part.find('"',left+1)
                if right<0:
                    return (None,orig)
                try:
                    line = json.loads('['+part[:right+1].strip()+']')[0]
                    break
                except:
                    left = right
            tail = part[right+1:]+text[stop:]
        return (line,head+tail)

    def parse_mergeseq(seq):
        res = None
        seq = seq.lstrip()
        while True:
            left = seq[0:5]
            if left=="<'EM_":
                right = "'>="
            elif left=="{'EM_":
                right = "'}="
            else:
                return res
            stop = seq.find(right)
            if stop<1:
                return res
            what = seq[0:stop+2]
            seq = seq[stop+3:]
            left = seq[0:2]
            if left=="<'":
                right = '>, '
            elif left=="{'":
                right = '}, '
            else:
                return res
            stop = min_or_all(seq.find(right+"<'"),seq.find(right+"{'"),len(seq))
            repl = seq[0:stop+1]
            seq = seq[stop+3:]
            if res is None:
                res = {}
            res[what] = repl
    
    def min_or_all(a,b,n):
        if a>=0:
            if b>=0:
                if a<b:
                    return a
                return b
            else:
                return a
        elif b>=0:
            return b
        return n
        
    def dict_replace(di,text):
        for key in di:
            text = text.replace(key,di[key])
        return text

    gr_lock = threading.Lock()
    
    def gr_func(gr_name,gr_text,gr_radio,store):
        with gr_lock:
            gr_orig = gr_text
            font = 'font-family:Consolas,Courier New,Courier,monospace;'
            table = '<style>.webui_embedding_merge_table,.webui_embedding_merge_table td,.webui_embedding_merge_table th{border:1px solid gray;border-collapse:collapse}.webui_embedding_merge_table td,.webui_embedding_merge_table th{padding:2px 5px;text-align:center;vertical-align:middle;'+font+'font-weight:bold;}</style><table class="webui_embedding_merge_table">'
            (reparse,request) = parse_infotext(gr_text)
            if reparse is not None:
                reparse = parse_mergeseq(reparse)
                if reparse is None:
                    return ('<center><b>Prompt restore failed!</n></center>',gr_name,gr_orig)
                else:
                    request = dict_replace(reparse,request)
                    return ('<center><b>Prompt restored.</n></center>',gr_name,request)
            if gr_text[:1]=="'":
                clipskip = opts.CLIP_stop_at_last_layers
                opts.CLIP_stop_at_last_layers = 1
                (res,err) = gr_parser(gr_text)
                opts.CLIP_stop_at_last_layers = clipskip
                if (res is not None) and res.numel()==0:
                    err = 'Result is ZERO vectors!'
                if err is not None:
                    txt = '<b style="'+font+'">'+html.escape(err)+'</b>'
                else:
                    txt = table+'<tr><th>Index</th><th>Min</th><th>Max</th><th>Sum</th><th>Abs</th><th>Len</th>'
                    i = 1
                    for one in res:
                        txt += '<tr><td>{}</td>{}</tr>'.format(i,tensor_info(one))
                        i += 1
                    txt += '<tr><td colspan="6">&nbsp;</td></tr>'
                    txt += '<tr><td>ALL:</td>{}</tr>'.format(tensor_info(res))
                    txt += '</table>'
                return ('<center>'+txt+'</center>',need_save_embed(store,gr_name,res),gr_orig)
            if gr_text.find("<'")>=0 or gr_text.find("{'")>=0:
                cache = reset_temp_embeddings(False)
                used = {}
                (res,err) = merge_one_prompt(cache,{},{},used,gr_text,False,False)
                if err is not None:
                    txt = '<b style="'+font+'">Embedding Merge failed - '+html.escape(err)+'</b>'
                    return ('<center>'+txt+'</center>',gr_name,gr_orig)
                gr_text = res
            by_none = 0
            by_comma = 1
            by_parts = 2
            by_words = 3
            by_tokens = 4
            by_vectors = 5
            tok2txt = tokens_to_text()
            clipskip = opts.CLIP_stop_at_last_layers
            opts.CLIP_stop_at_last_layers = 1
            if gr_radio!=by_comma:
                res = text_to_vectors(gr_text)
                if (gr_radio==by_none) and (res is not None) and (len(res)!=0):
                    res = [res]
            else:
                res = []
                split = gr_text.split(',')
                for part in split:
                    one = text_to_vectors(part.strip())
                    if one:
                        res.append(one)
                    else:
                        res = None
                        break
            opts.CLIP_stop_at_last_layers = clipskip
            if (res is None) or (len(res)==0):
                if gr_text.strip()=='':
                    return ('',gr_name,gr_orig)
                txt = '<b>Failed to parse! (Possibly there are more than 75 tokens; or extra spaces inside embed names). Embeddings are not shown now:</b><br/><br/>'
                tokens = text_to_tokens(gr_text)
                if tokens:
                    txt += table+'<tr><th>Index</th><th>Vectors</th><th>Text</th><th>Token</th></tr>'
                    if tok2txt:
                        pairs = tok2txt(tokens)
                    else:
                        pairs = [([tok],'<ERROR>') for tok in tokens]
                    index = 1
                    for arr, text in pairs:
                        length = len(arr)
                        if length==0:
                            continue
                        txt += '<tr><td>'+(str(index) if length==1 else str(index)+'-'+str(index+length-1))+'</td><td>'+str(length)+'</td><td>'+html.escape('"'+text+'"')+'</td><td>'+(', '.join([str(a) for a in arr]))+'</td></tr>'
                        index += length
                    txt += '</table>'
                return ('<center>'+txt+'</center>',gr_name,gr_orig)
            txt = table+'<tr><th>Index</th><th>Vectors</th><th>Text</th><th>Token</th><th>Min</th><th>Max</th><th>Sum</th><th>Abs</th><th>Len</th></tr>'
            index = 1
            join = False
            if gr_radio==by_words:
                join = True
                gr_radio = by_tokens
            elif (gr_radio==by_none) or (gr_radio==by_comma):
                r_res = []
                for one in res:
                    r_tensor = []
                    r_name = ''
                    r_tokens = []
                    for tensor, name, tokens in one:
                        r_tensor.append(tensor)
                        if tok2txt and tokens and gr_radio==by_none:
                            split = tok2txt(tokens)
                            name = ''
                            tokens = []
                            for s_tokens, s_name in split:
                                name += s_name
                                tokens += s_tokens
                        r_name += name
                        if tokens:
                            r_tokens += tokens
                        else:
                            r_tokens += ['*_'+str(tensor.size(0))]
                            if gr_radio==by_none:
                                r_name += ' '
                    r_res.append((torch.cat(r_tensor),r_name,r_tokens))
                res = r_res
                gr_radio = by_parts
            for tensor, name, tokens in res:
                split = None
                size = tensor.size(0)
                span = ''
                if gr_radio!=by_parts:
                    span = ' rowspan="'+str(size)+'"'
                    if tokens and tok2txt:
                        split = tok2txt(tokens)
                        if join:
                            comb = []
                            last = -1
                            for s_arr, s_text in split:
                                if (last<0) or (comb[last][1][-1:]==' '):
                                    comb.append((s_arr,s_text))
                                    last += 1
                                else:
                                    comb[last] = (comb[last][0]+s_arr,comb[last][1]+s_text)
                            split = comb
                    if gr_radio==by_tokens:
                        if split is not None:
                            span = ' rowspan="'+str(len(split))+'"'
                        else:
                            span = ''
                head = '<td'+span+'>'+(str(index) if size==1 else str(index)+'-'+str(index+size-1))+'</td><td'+span+'>'+str(size)+'</td>'
                if split is None:
                    head += '<td'+span+'>'+html.escape('"'+name+'"')+'</td>'
                index += size
                if (gr_radio==by_vectors) or ((gr_radio==by_tokens) and (tokens is not None)):
                    i = 0
                    part = 0
                    j = 0
                    ten = None
                    column = ''
                    toks = None
                    for one in list(tensor):
                        i += 1
                        use = one
                        if split is not None:
                            if part==0:
                                pair = split[j]
                                part = len(pair[0])
                                if gr_radio==by_tokens:
                                    column = '<td>'+html.escape('"'+pair[1]+'"')+'</td>'
                                    toks = ', '.join([str(t) for t in pair[0]])
                                else:
                                    column = '<td rowspan="'+str(part)+'">'+html.escape('"'+pair[1]+'"')+'</td>'
                                j += 1
                        part -= 1
                        if gr_radio==by_tokens:
                            if ten==None:
                                ten = []
                            ten.append(one)
                            if part>0:
                                continue
                            use = torch.stack(ten)
                            tok = toks if tokens else '*'
                        else:
                            tok = tokens[i-1] if tokens else '*_'+str(i)
                        txt += '<tr>{}{}<td>{}</td>{}</tr>'.format(head,column,tok,tensor_info(use))
                        column = ''
                        head = ''
                        ten = None
                else:
                    txt += '<tr>{}<td>{}</td>{}</tr>'.format(head,', '.join([str(t) for t in tokens]) if tokens else '*',tensor_info(tensor))
            txt += '</table>'
            return ('<center>'+txt+'</center>',need_save_embed(store,gr_name,res),gr_orig)

    def tensor_info(tensor):
        return '<td>{:>-14.8f}</td><td>{:>+14.8f}</td><td>{:>+14.8f}</td><td>{:>14.8f}</td><td>{:>14.8f}</td>'.format(tensor.min().item(),tensor.max().item(),tensor.sum().item(),tensor.abs().sum().item(),torch.linalg.norm(tensor,ord=2)).replace(' ','&nbsp;')

    merge_dir = None
    
    def need_save_embed(store,name,vectors):
        if not store:
            return name
        name = ''.join( x for x in name if (x.isalnum() or x in '._- ')).strip()
        if name=='':
            return name
        try:
            if type(vectors)==list:
                vectors = torch.cat([r[0] for r in vectors])
            file = modules.textual_inversion.textual_inversion.create_embedding('_EmbeddingMerge_temp',vectors.size(0),True,init_text='')
            pt = torch.load(file,map_location='cpu')
            token = list(pt['string_to_param'].keys())[0]
            pt['string_to_param'][token] = vectors.cpu()
            torch.save(pt,file)
            target = os.path.join(merge_dir,name+'.pt')
            os.replace(file,target)
            modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
            return ''
        except:
            traceback.print_exc()
            return name

    def embedding_merge_dir():
        try:
            nonlocal merge_dir
            merge_dir = os.path.join(cmd_opts.embeddings_dir,'embedding_merge')
            modules.sd_hijack.model_hijack.embedding_db.add_embedding_dir(merge_dir)
            os.makedirs(merge_dir)
        except:
            pass

    def raise_sd_error(p,msg):
        class Exception_From_EmbeddingMergeExtension_():
            def __getattribute__(self,_):
                raise Exception_From_EmbeddingMergeExtension(msg)
        p.__class__ = Exception_From_EmbeddingMergeExtension_

    def merge_one_prompt(cache,texts,parts,used,prompt,prod,only_count):
        try:
            if only_count:
                clip = modules.sd_hijack.model_hijack.clip
            cnt = 0
            if (prompt is None) or (prompt==''):
                return (prompt,None) if not only_count else (0,None)
            if prompt in texts:
                return (texts[prompt],None)
            orig = prompt
            left = 0
            while True:
                curly = prompt.find("{'",left)
                left = prompt.find("<'",left)
                if (curly>=0 and curly<left) or (left<0):
                    left = curly
                    curly = True
                else:
                    curly = False
                if left<0:
                    if only_count:
                        _, token_count = clip.process_texts([prompt])
                        cnt += token_count
                        prompt = cnt
                    texts[orig] = prompt
                    return (prompt,None)
                right = left
                while True:
                    right = prompt.find('}' if curly else '>',right+1)
                    if right<0:
                        if curly:
                            return (None,'Not found closing "}" after "{\'"')
                        else:
                            return (None,'Not found closing ">" after "<\'"')
                    if (prompt.count("'",left,right)&1)==0:
                        break
                part = prompt[left+1:right].strip()
                if part in parts:
                    embed = parts[part]
                else:
                    (res,err) = gr_parser(part)
                    if err is not None:
                        return (None,err)
                    if only_count:
                        if (res is None) or (res.numel()==0):
                            embed = 0
                        else:
                            embed = res.size(0)
                    else:
                        if (res is None) or (res.numel()==0):
                            embed = ''
                        else:
                            embed = add_temp_embedding(res,cache,prod,curly)
                            used[embed] = part
                parts[part] = embed
                if only_count:
                    _, token_count = clip.process_texts([prompt[:left]])
                    cnt += token_count+embed
                    prompt = prompt[right+1:]
                    left = 0
                else:
                    prefix = prompt[:left].rstrip()+' '+embed
                    left = len(prefix)
                    prompt = prefix+' '+(prompt[right+1:].lstrip())
        except:
            traceback.print_exc()
            return (None,'Fatal error?')

    def embedding_merge_extension(p):
        cache = reset_temp_embeddings(True)
        texts = {}
        parts = {}
        used = {}
        arr = [
            p.all_prompts,
            p.prompt if type(p.prompt)==list else [p.prompt],
            p.all_negative_prompts,
            p.negative_prompt if type(p.negative_prompt)==list else [p.negative_prompt],
        ]
        for one in arr:
            if one is not None:
                for i in range(len(one)):
                    (res,err) = merge_one_prompt(cache,texts,parts,used,one[i],True,False)
                    if err is not None:
                        raise_sd_error(p,'\n\nEmbedding Merge failed - '+err+'\n')
                        return
                    one[i] = res
        p.all_prompts = arr[0]
        p.all_negative_prompts = arr[2]
        p.prompt = arr[1] if type(p.prompt)==list else arr[1][0]
        p.negative_prompt = arr[3] if type(p.negative_prompt)==list else arr[3][0]
        gen = ''
        for embed in used:
            if embed[0]=='<':
                gen += embed+'=<'+used[embed]+'>, '
            else:
                gen += embed+'={'+used[embed]+'}, '
        if gen!='':
            p.extra_generation_params['EmbeddingMerge'] = gen[:-2]

    try:
        cls = modules.sd_hijack.StableDiffusionModelHijack
        field = '__embedding_merge_wrapper'
        def hook_prompt_lengths(self,text):
            if text.find("<'")<0 and text.find("{'")<0:
                return get_prompt_lengths(self,text)
            (cnt,err) = merge_one_prompt(None,{},{},None,text,True,True)
            print(cnt,err)
            if err is not None:
                return -1,-1
            return cnt, self.clip.get_target_prompt_token_count(cnt)
        if hasattr(cls,field):
            get_prompt_lengths = getattr(cls,field)
        else:
            get_prompt_lengths = cls.get_prompt_lengths
            setattr(cls,field,get_prompt_lengths)
        cls.get_prompt_lengths = hook_prompt_lengths
    except:
        traceback.print_exc()

    def on_infotext_pasted(infotext,result):
        if 'EmbeddingMerge' in result:
            reparse = result['EmbeddingMerge']
            if reparse[:1]=='"':
                try:
                    reparse = json.loads('['+reparse.strip()+']')[0]
                    reparse = parse_mergeseq(reparse)
                except:
                    reparse = None
            else:
                reparse = parse_mergeseq(reparse)
            request = None
        else:
            (reparse,request) = parse_infotext(infotext)
            if reparse is not None:
                reparse = parse_mergeseq(reparse)
        if reparse is not None:
            if 'Prompt' in result:
                if (request is not None) and (result['Prompt']==infotext):
                    result['Prompt'] = request
                result['Prompt'] = dict_replace(reparse,result['Prompt'])
            if 'Negative prompt' in result:
                result['Negative prompt'] = dict_replace(reparse,result['Negative prompt'])

    global _webui_embedding_merge_extension_
    _webui_embedding_merge_extension_ = embedding_merge_extension
    embedding_merge_dir()
    
    setattr(_webui_embedding_merge_,'on_infotext_pasted',on_infotext_pasted)
    
    return gr_tab

script_callbacks.on_ui_tabs(_webui_embedding_merge_())
script_callbacks.on_infotext_pasted(_webui_embedding_merge_.on_infotext_pasted)

#EOF