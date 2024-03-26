# stable-diffusion-webui-embedding-merge

'''
WebUI Dependencies:

1) Class <modules.textual_inversion.textual_inversion.Embedding> is used to create embeddings.
     required fields: <.vec> = actual tensor, <.vectors> = first dim size, <.shape> = last dim size.
2) Object <modules.sd_hijack.model_hijack.embedding_db> is abused to create ephemeral embeddings.
     Work with fields <.word_embeddings> and <.ids_lookup> is replicated from
     </modules/textual_inversion/textual_inversion.py>, refer to <register_embedding()> here.
     UPD: not needed anymore, since upstream implemented <register_embedding_by_name()>
3) Saving of embeddings is done by crafting a proper shape for .pt file manually, and then
     <modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload)> is called.
4) <modules.sd_hijack.StableDiffusionModelHijack.get_prompt_lengths(text)> is hooked but not replaced.
5) Part of <encode_embedding_init_text()> from <sd_hijack_clip.py> and <sd_hijack_open_clip.py> is converted to
     <tokens_to_vectors()> here; it uses <shared.sd_model.cond_stage_model.wrapped> and then calls either
     <.model.token_embedding.wrapped()> for SD2, or <.transformer.text_model.embeddings.token_embedding.wrapped()> for SD1
6) Code from <https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer> is heavily copied:
     it grabs <shared.sd_model.cond_stage_model.wrapped> and checks it against
     <FrozenCLIPEmbedder> and <FrozenOpenCLIPEmbedder>, refer to <tokens_to_text()> here.
7) <shared.sd_model.cond_stage_model.tokenize_line(line)> is called many times when parsing prompts.
     The code is very dependent on what it returns! Source in </modules/sd_hijack_clip.py>
     Also <shared.sd_model.cond_stage_model.tokenize()> can be called.
8) Method <p.cached_params()> is faked to be always unique if any runtime embeddings are detected to prevent wrong caching.
'''

import re
import os
import torch
import json
import html
import time
import types
import traceback
import threading
import gradio
import modules
from modules import shared, scripts, script_callbacks, devices, processing
from modules.shared import opts, cmd_opts
from modules.textual_inversion.textual_inversion import Embedding
import open_clip.tokenizer

def _webui_embedding_merge_():

    class Exception_From_EmbeddingMergeExtension(Exception):
        pass
    class Exception_From_EmbeddingMergeExtension_():
        def __init__(self,_):
            self._ = _
        def __getattr__(self,_):
            raise Exception_From_EmbeddingMergeExtension(self._)

    def gr_tab():
        with gradio.Blocks(analytics_enabled=False) as block:
            gradio.HTML('<style>#tab_embedding_merge_extension p::before,#tab_embedding_merge_extension p::after,#tab_embedding_merge_extension code::before,#tab_embedding_merge_extension code::after{display:none!important}</style>')
            with gradio.Row():
                with gradio.Accordion('Embedding Merge extension! (Click here for usage instructions)', open=False):
                    with gradio.Accordion('Introduction...', open=False):
                        gradio.Markdown('''
## Purpose:

Did you know that StableDiffusion reads your prompt by so-called tokens? They are multidimensional numerical vectors that construct together words and phrases.

It is actually possible to create new words by simple merging (adding) different vectors together, resulting in something that could mean both things simultaneously!

However, it is not always working, and sometimes it won't give what you would expect, but it is definitely worth experimenting.

Basically, this extension will create Textual Inversion embeddings purely by token merging (without any training on actual images!) either automatically during generation, or manually on its tab.

## Usage:

The tab `EM` can be used to:
- inspect your prompt or specific words
- create TI embeddings from text fragments with or without merging
- check correctness of your merge expressions
''')
                    gradio.Markdown('''
### TL;DR:

Use syntax `<'one thing'+'another thing'>` to merge terms "one thing" and "another thing" together in one single embedding in your positive or negative prompts at runtime.

Also use `<'your words'*0.5>` (or any number, default is 1.0) to increase or decrease the essence of "your words" (which can be even zero to disable that part of the prompt).

To use attention with round brackets ( ), put them around < >, like `(<'one'+'two'>:0.9)`  
Use as many <> in one prompt, as you want; also you can put your existing TI embedding names inside `' '`.

~~When you need literal <' for some reason, put a space between.~~ You cannot have literal <' anywhere in your prompts; but with a space between (`< '`) it will be ignored by this extension.

If some other extension interferes with this syntax, change angular brackets to curly: `{'also works'*4}`

## View text or embeddings vectors

You can paste your vanilla prompt (without any other special syntax) into the textbox in EM tab to see how it is parsed by WebUI. All of detected Textual Inversion embeddings will be extracted and presented to you along with literal text tokens. For example:

>intergalactic train, masterpiece, by Danh Víµ
''')
                    with gradio.Accordion('More about table columns and grouping of its rows...', open=False):
                        gradio.Markdown('''
### Rows:

- `By none` = interpret the prompt as a whole, extracting all characters from real tokens
- `By comma` = split the prompt by tags on commas, removing commas but keeping source space characters
- `By parts` (default) = split at TI embeddings, joining text parts together, keeping spaces
- `By words` = split only after tokens that actually produce space character at the end
- `By tokens` = split at everything except characters that are represented with more than one vector
- `By vectors` = show all vectors separated, even for TI embeddings

### Columns:

- `Index` = index of one vector or index range (inclusive) for this row
- `Vectors` = number of final vectors for this row (to clearly see it)
- `Text` = original or recreated from tokens text, enclosed in quotes for clarity
- `Token` = list of CLIP token numbers that represent this row; for TI embeddings \* or \*_X where X is the index of current embedding vector
- `Min` = lowest (negative) value of the vector or grouped vectors values
- `Max` = largest value
- `Sum` = sum of all values with sign
- `Abs` = sum of modulus of each value, without sign (always positive)
- `Len` = vector length in L2 norm, square root of sum of squared values (computed approximate)
- `Std` = standard deviation for vector values.

### Why do you need it:

To make sure your prompt is interpreted the way you expect (for example, that existing TI embeddings are detected). Also you can explore CLIP tokens this way.

If you type a new name into the textbox on the bottom, your whole current prompt will be converted into a single Textual Inversion embedding with that name (and stored inside `/embeddings/embedding_merge/` subdirectory). You can use this for:

- Creating a shortened part to quickly use in prompts (not recommended though, since you will lose the original text later), but with no other benefits;
- Prepare TI embedding for actual training by using existing embeddings for its initialization.
''')
                    gradio.Markdown('''
## Test merge expression:

In EM tab you can enter a "merge expression" that starts with a single quote, to see how it will be parsed and combined by this extension. It should contain single quotes around literal texts or TI embeddings, and special operators between them. For example:

>'greg rutkowski'/4+'gustav dore'*0.75
''')
                    with gradio.Accordion('More about merge expression syntax...', open=False):
                        gradio.Markdown('''
### Expression syntax:

- `'one' + 'two'` = blend vectors together by simple sum of all values. If length is different, smallest part will be right-padded with zeroes.
- `'one' - 'two'` = as above, but subtraction. Note that + and - can be put only between textual parts and will have lowest priority.
- `'text' * NUM` = multiply all vectors of quoted literal by numeric value. You can use floating point (0.85) and negative numbers (-1), but not arithmetic expressions.
- `'text' / NUM` = division by number, just as multiplication above. Applies to previous text literal but after previous similar operations, so you can multiply and divide together (\*3/5)
- `'text' : NUM` = change vector count of literal, to shrink or enlarge (padded with zeros). Only integer without sign!
- `'text' :+ NUM` and `'text'  :- NUM` = circular rotate vectors in this token, for example +1 will shift index of each vector by one forward, wrapping on last.
- `'text',NUM` (chainable as `'a',B,'c','d',E,F…`) = concatenate text with a token by its numerical index (so, to get any pure token – use empty left string: `'',256`). Special tokens: `0000` = "start token" (index 49406), `000` = "end token" (index 49407), `00` = "padding token" (also 49407 for SD1, but 0 for SD2). Token number `0` is not zero-vector, but for some reason counts as symbol "!" without a space after it, which is impossible to normally enter anyway.

To apply multiplication (or division), cropping or shifting **to the result** of addition (or subtraction), you cannot use parenthesis; instead, try this syntax:

- `'one' + 'two' =* NUM` = will multiply the sum of 'one' and 'two', but not 'two' alone
- `'one' + 'two' =/ NUM` = divide the sum (or any number of sums to the left), effectively the "result" of everything
- `'one' + 'two' =: NUM` = crop or enlarge the results
- `'one' + 'two' =:+ NUM` or `'one' + 'two' =:- NUM` = rotate the result

Thus, the following operations are doing the same:

>`'a'/2 + 'b'/2 + '':1 - 'd'`  
`'a'+'b' =* 0.5 + 'c'*0 + 'd'*-1`

There is no true "concatenation" operator (since you will be able to concatenate several separate merge expressions later), but you may replicate it with addition of the same text enlarged and shifted, if you need.  
Operation "," has the highest priority (it will directly construct the string before doing anything else), so you cannot concatenate anything to the result of addition or multiplication. Use it only to add tokens by index in your text.

For example, repeating a two-vector word, resulting in 4 vectors of two equal pairs:

> 'artstation' + 'artstation' :4 :+2  
> 'artstation','artstation'

You can use shifting to join several vectors of the same text together. For example, given a 4-vectors word you may merge those vectors in one:

> 'kuvshinov' + 'kuvshinov':-1 + 'kuvshinov':-2 + 'kuvshinov':-3 =: 1  
> '',1836 + '',85 + '',43074 + '',341

Note that those indices are referring to "ku|v|shino|v[space]" and cannot be entered from raw text, since it would be parsed as "ku[space]", "v[space]" and "shino[space]", which are different tokens!

When you merge strings of unequal length, shortest one is padded with zero vectors; if you want to pad it with something else, you should check the vector count and concatenate accordingly:

> 'close-up',00,00 + 'out-of-frame' + 'cropped',00,00,00,00  
> 'up',00,00+'of-frame'+'',00,00,00 =:5:+2 + 'close-'+'out-'+'cropped',00

### Why do you need it:

To prepare your expression and fix any errors. You can evaluate its correctness by roughly comparing numbers in table (for example, adding vectors will generally result in higher `Abs` value; while multiplication is directly changing all numbers straightforwardly).

If for some reason you couldn't use the syntax for merging prompts at runtime, at least you will be able to enter a name and create a regular TI embedding from your merge expression. Then you may use it even without this extension installed!

Also you can check numerical parameters of your trained textual embedding and compare it with "normal" vectors. For example, very large `Len` or `Std` will mean that something is wrong and at least you may divide it in attempt to fix.
''')
                    gradio.Markdown('''
## Several merge expressions in prompt:

If you put a valid merge expression enclosed in angular <'…' …> or curly {'…' …} brackets anywhere in your prompt (with no space between `<` or `{` and `'`) on EM tab, it will be parsed and merged into one temporary Textual Inversion embedding, which replaces the expression itself. The resulting prompt will be joined from those embeddings and anything between expressions. For example:

>A photo of <'cat'+'dog'>, {'4k'+'dynamic lighting'+'science fiction'=/3} masterpiece
''')
                    with gradio.Accordion('More examples of using angular/curly brackets...', open=False):
                        gradio.Markdown('''
### More examples:


Combining different subjects or styles together, resulting in joined concepts:

> A realistic photo of the <'girl'+'doll'> in rainbow dress standing on a shore.  
Art by <'greg rutkowski'*X+'hayao miyazaki'*Y> style.

Notes:
- Works best when all of your subjects have the same number of vectors (also can be roughly simulated by BREAK statement: `… photo of the girl in rainbow … BREAK … photo of the doll in rainbow …`);
- You don't have to divide on the number of added parts, especially if your subjects are very different (e.g. not contain same tokens);
- By multiplying each part in second example (where X and Y are numbers between 0.0 and 1.0) you may get a weighed combination or interpolation.

Changing weight of individual words in prompt:

> A <'peacock'*X> is standing on a top of <'giraffe'*Y>.  
worst quality, ugly, <'bad anatomy,':0> blurry, cropped

Where X and Y will be numbers from 0.0 to 1.0 or even higher, up to 5. This way you can directly change relative affection between subjects.

Notes:
- Often values between 0.5 and 1.5 don't really change anything, looking like plain 1.0
- Values lower than 0.5 and near to 0.0 are greatly reducing subject weight indeed! Up to its complete absence (which is not possible otherwise, for example even zero attention `(word:0)` does not eliminate "word" from the prompt)
- High numbers might increase the presence of an object, not in quantity but in essence. Very high multipliers (above 10) corrupt the subject, but still don't destroy the image itself.

Eliminating a part of the negative prompt by zeroing its vectors can be used to understand the effect of the part in question, without shifting the rest of the text otherwise. Since WebUI is splitting long prompts at arbitrary commas (and then merging resulting parts together), simple deletion of a part might change things severely.
''')
                    gradio.Markdown('''
## Using merge expressions in prompts at runtime!

You can actually put merge expressions in angular or curly brackets into your txt2img or img2img prompt in WebUI. This extension will intercept both main and negative prompts, parse and merge expressions creating temporary TI embeddings that WebUI will "see" instead of your original text. In generation info there will be internal meaningless names like <'EM_1'>, but extra parameter "EmbeddingMerge" will contain original merge expressions. To quickly restore your prompts, just paste your complete generation information (from .txt or PNG Info) into the textbox on EM tab (also it should work for the official "paste" toolbar button too) – its temporary embeddings will be replaced back with expressions, for example:

> a photo of <'EM_1'>  
Negative prompt: {'EM_2'}  
Steps: 8, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1374372309, Size: 512x512, Model hash: c6bbc15e32, Model: sd-v1-5-inpainting, EmbeddingMerge: "<'EM_1'>=<'sky' * 2/4 + 'forest' * 3/4>, {'EM_2'}={'blurry'+'cropped'}", Conditional mask weight: 1

For your information replicating start tokens of the syntax itself:
- `<'` = `<'',27,6>` or `<'',27,262>`
- `{'` = `<'',90,6>` or `<'',90,262>`

''')
                    with gradio.Accordion('Limitations...', open=False):
                        gradio.Markdown('''
### What is not working:

#### Binding properties to objects:

> Photo of a <'blonde'+'boy'> in <'red'+'shirt'> wearing <'green'+'pants'> and <'blue'+'shoes'>

– results in anything but not what was requested.

#### Collapsing artists to single token:

> Painting by <'William' + '-' + 'Adolphe'+'Adolphe':+1 + 'Bouguereau'+'Bouguereau':+1+'Bouguereau':+2 =:1>. A girl, masterpiece

– results in something barely distinct from zeroing the term altogether.

#### Subtracting concepts as in word2vec:

> Full-body photo of a <'king'-'man'+'woman'>  
Detailed photo of <'yellow'-'red'> car

– generally results in totally ruined composition.

#### Simulating negative prompt via negation of words:

> A portrait of the princess. <'frame, black-white'*-1>  
A cat is chasing a dog. <''-'road'-'grass'>

– will still add those concepts to positive prompt, but with weird presence. You could find more luck with small values `-0.1-0.0` though.
''')
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
            clip = shared.sd_model.cond_stage_model
            if hasattr(clip,'embedders'):
                clip = clip.embedders[0]
            clip = clip.wrapped
            typename = type(clip).__name__.split('.')[-1]
            if typename=='FrozenOpenCLIPEmbedder':
                clip = OpenClip(clip)
            else:
                clip = VanillaClip(clip)
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

    def get_model_clips():
        clip = shared.sd_model.cond_stage_model
        if(hasattr(clip,'embedders')):
            try:
                return (clip.embedders[0],clip.embedders[1]) # SDXL
            except:
                pass
        return (clip,) # SD1 or SD2

    def text_to_vectors(orig_text):
        try:
            both = []
            for clip,lg in zip(get_model_clips(),('clip_l','clip_g')):
                res = []
                text = orig_text.lstrip().lower()
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
                    if type(tensor)==dict:
                        tensor = tensor[lg]
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
                both.append(res)
            return both
        except:
            traceback.print_exc()
            return None

    def text_to_tokens(text):
        try:
            both = []
            for clip in get_model_clips():
                tokens = clip.tokenize([text])[0]
                both.append(tokens)
            if len(both)>1:
                if (both[0]-both[1]).abs().max().item() != 0:
                    print('EM: text_to_tokens',both)
                    return None
            return both[0]
        except:
            return None

    def tokens_to_vectors(pair):
        try:
            res = []
            for clip,arr in zip(get_model_clips(),pair):
                clip = clip.wrapped
                if hasattr(clip,'model') and hasattr(clip.model,'token_embedding'):
                    tensor = torch.tensor([arr],dtype=torch.int,device=devices.device)
                    tokens = clip.model.token_embedding.wrapped(tensor).to(devices.device)
                else:
                    token_embedding = clip.transformer.text_model.embeddings.token_embedding
                    tensor = torch.tensor([arr],dtype=torch.int,device=token_embedding.wrapped.weight.device)
                    tokens = token_embedding.wrapped(tensor).to(devices.device)
                res.append(tokens)
            if len(res)>1:
                if len(res[0]) != len(res[1]):
                    print('EM: tokens_to_vectors',res)
                    return None
            return res
        except:
            traceback.print_exc()
            return None

    def to_float(num):
        if num is None:
            return None
        try:
            return float(num)
        except:
            return None

    def to_int(num):
        if num is None:
            return None
        try:
            return int(num)
        except:
            return None

    def grab_vectors(text):
        try:
            both = []
            for res in text_to_vectors(text):
                if res is None:
                    return None
                if len(res)==0:
                    res = text_to_vectors(',')[len(both)][0][0][0:0]
                else:
                    res = torch.cat([ten[0] for ten in res]);
                both.append(res)
            if len(both)>1:
                if len(both[0]) != len(both[1]):
                    print('EM: grab_vectors',both)
                    return None
            return both
        except:
            return None

    reg_clean = re.compile(r'\s+')
    reg_oper = re.compile(r'(=?)(?:([*/,])([+-]?[0-9]*(?:\.[0-9]*)?)|:([+-]?)(-?[0-9]+))')

    def merge_parser(text,only_count):
        clips = get_model_clips()
        vocab = None
        def check_vocab(token2):
            nonlocal vocab
            if vocab is None:
                vocab = []
                for clip in clips:
                    wrapped = clip.wrapped
                    typename = type(wrapped).__name__.split('.')[-1]
                    if typename=='FrozenCLIPEmbedder':
                        voc = wrapped.tokenizer.get_vocab()
                    elif typename=='FrozenOpenCLIPEmbedder':
                        voc = open_clip.tokenizer._tokenizer.encoder
                    else:
                        return True
                    vocab.append({v: k for k, v in voc.items()})
            t = token2[0]
            if len(vocab)>1:
                if len(token2)>1:
                    return (t in vocab[0]) and (token2[1] in vocab[1])
                return (t in vocab[0]) and (t in vocab[1])
            return t in vocab[0]
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
        combine = False
        for param, quot in arr:
            one = param
            if quot:
                if combine:
                    actions[-1]['V'] = param
                    combine = False
                else:
                    actions.append({
                      'A': None,
                      'V': param,
                      'O': one,
                    })
                continue
            elif combine:
                return (None,'Wrong concatenation "'+param+'" in '+orig)
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
                m_tok = -1
                if m_val is not None:
                    if m_mul==',':
                        if m_flag:
                            return (None,'Concatenation doesn\'t support \'=\' prefix: "'+param+'" in '+orig)
                        if (len(m_val)>0) and (m_val[0]=='0'):
                            if m_val=='0':
                                m_tok = 0
                            elif m_val=='00':
                                m_tok = -2
                            elif m_val=='000':
                                m_tok = -3
                            elif m_val=='0000':
                                m_tok = -4
                            else:
                                m_tok = None
                        elif m_val=='':
                            m_tok = -5
                            combine = True
                            m_val = None
                        else:
                            m_tok = to_int(m_val)
                            if (m_tok is not None) and not (m_tok>=0):
                                m_tok = None
                        if m_tok is None:
                            return (None,'Bad param for concatenation "'+param+'" in '+orig)
                    else:
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
                  'T': m_tok,
                  'O': one,
                })
                param = param[len(m.group(0)):]
        if combine:
            return (None,'Unfinished concatenation in '+orig)
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
                    if only_count:
                        if left>right:
                            right = left
                    else:
                        (vectors1_0,length1_0) = left[0].size()
                        (vectors2_0,length2_0) = right[0].size()
                        (vectors1_1,length1_1) = left[1].size() if len(left)>1 else (vectors1_0,length1_0)
                        (vectors2_1,length2_1) = right[1].size() if len(right)>1 else (vectors2_0,length2_0)
                        if (length1_0!=length2_0) or (length1_1!=length2_1) or (vectors1_0!=vectors1_1) or (vectors2_0!=vectors2_1) or (len(left)!=len(right)):
                            return (None,'Cannot merge different embeddings in '+orig)
                        if vectors1_0!=vectors2_0:
                            if vectors1_0<vectors2_0:
                                target = [torch.zeros(vectors2_0,length1_0).to(device=devices.device,dtype=torch.float32)]
                                target[0][0:vectors1_0] = left[0]
                                if len(left)>1:
                                    target.append(torch.zeros(vectors2_1,length1_1).to(device=devices.device,dtype=torch.float32))
                                    target[1][0:vectors1_1] = left[1]
                                left = target
                            else:
                                target = [torch.zeros(vectors1_0,length2_0).to(device=devices.device,dtype=torch.float32)]
                                target[0][0:vectors2_0] = right[0]
                                if len(right)>1:
                                    target.append(torch.zeros(vectors1_1,length2_1).to(device=devices.device,dtype=torch.float32))
                                    target[1][0:vectors2_1] = right[1]
                                right = target
                        if add>0:
                            right[0] = left[0]+right[0]
                            if len(left)>1 and len(right)>1:
                                right[1] = left[1]+right[1]
                        else:
                            right[0] = left[0]-right[0]
                            if len(left)>1 and len(right)>1:
                                right[1] = left[1]-right[1]
                left = None
            A = act['A']
            if A==None:
                line = act['V']
                if line==None:
                    return (right,None)
                right = grab_vectors(line)
                if right==None:
                    return (None,'Failed to parse \''+line+'\' in '+orig)
                if only_count:
                    right = right[0].size(0)
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
                t = act['T']
                if only_count:
                    if t!=-1:
                        right += 1
                    elif (r==0)and(s>=0):
                        right = s
                else:
                    if t!=-1:
                        if t<0:
                            if t==-2:
                                t = [clip.id_pad for clip in clips]
                            elif t==-3:
                                t = [clip.id_end for clip in clips]
                            elif t==-4:
                                t = [clip.id_start for clip in clips]
                            else:
                                res = grab_vectors(act['V'])
                                t = None
                                if res is None:
                                    return (None,'Failed to parse \''+act['V']+'\' in '+orig)
                        else:
                            if len(clips)>1:
                                t = [t,t]
                            else:
                                t = [t]
                        if t is not None:
                            if not check_vocab(t):
                                return (None,'Unknown token value \''+str(t[0])+'\' in '+orig)
                            res = tokens_to_vectors(t)
                        if res is None:
                            return (None,'Failed to convert token \''+str(t)+'\' in '+orig)
                        if right is None:
                            right = res
                        else:
                            if len(right)>1 and len(res)>1:
                                right = [torch.cat([right[0],res[0]]),torch.cat([right[1],res[1]])]
                            else:
                                right = [torch.cat([right[0],res[0]])]
                    elif r!=0:
                        right[0] = right[0].roll(r,dims=0)
                        if len(right)>1:
                            right[1] = right[1].roll(r,dims=0)
                    else:
                        if s>=0:
                            (vectors,length) = right[0].size()
                            if vectors>s:
                                if len(right)>1:
                                    right = [right[0][0:s],right[1][0:s]]
                                else:
                                    right[0] = right[0][0:s]
                            elif vectors<s:
                                target = [torch.zeros(s,length).to(device=devices.device,dtype=torch.float32)]
                                target[0][0:vectors] = right[0]
                                if len(right)>1:
                                    target.append(torch.zeros(s,length).to(device=devices.device,dtype=torch.float32))
                                    target[1][0:vectors] = right[1]
                                right = target
                        elif act['W']==True:
                            right = [r*act['V'] for r in right]
                        elif  act['W']==False:
                            right = [r/act['V'] for r in right]
        return (right,None)

    def grab_embedding_cache():
        db = modules.sd_hijack.model_hijack.embedding_db
        field = '__embedding_merge_cache_'
        if hasattr(db,field):
            cache = getattr(db,field)
        else:
            cache = {'_':0,'-':0,'/':0}
            setattr(db,field,cache)
        return cache

    def register_embedding(name,embedding):
        self = modules.sd_hijack.model_hijack.embedding_db
        model = shared.sd_model
        if hasattr(self,'register_embedding_by_name'):
            return self.register_embedding_by_name(embedding,model,name)
        # /modules/textual_inversion/textual_inversion.py
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

    def make_temp_embedding(name,vectors,cache,fake):
        embed = None
        if name in cache:
            embed = cache[name]
            if fake>0:
                return
        else:
            if fake>0:
                if len(get_model_clips())>1:
                    vectors = [torch.zeros((fake,16)),torch.zeros((fake,16))]
                else:
                    vectors = [torch.zeros((fake,16))]
        shape = vectors[-1].size()
        if len(vectors)>1:
            vectors = {'clip_g':vectors[1],'clip_l':vectors[0]}
        else:
            vectors = vectors[0]
        if embed is None:
            embed = Embedding(vectors,name)
            cache[name] = embed
        embed.vec = vectors
        embed.step = None
        embed.vectors = shape[0]
        embed.shape = shape[-1]
        embed.cached_checksum = None
        embed.filename = ''
        register_embedding(name,embed)

    def reset_temp_embeddings(prod,unregister):
        cache = grab_embedding_cache()
        num = cache[prod]
        cache[prod] = 0
        for a,b in (('<','>'),('{','}')):
            i = num
            while i>0:
                tgt = a+"'EM"+prod+str(i)+"'"+b
                if tgt in cache:
                    embed = cache[tgt]
                    if type(embed.vec)==dict:
                        for k,v in embed.vec.items():
                            embed.vec[k] = torch.zeros((0,v.shape[-1]),device=v.device)
                    else:
                        embed.vec = torch.zeros((0,embed.vec.shape[-1]),device=embed.vec.device)
                    embed.vectors = 0
                    embed.cached_checksum = None
                    del cache[tgt]
                    if unregister:
                        register_embedding(tgt,None)
                i = i-1
        return cache

    def add_temp_embedding(vectors,cache,prod,curly,fake):
        if fake>0:
            prod = '/'
            num = (cache[prod] or 0)
            if fake>num:
                cache[prod] = fake
            num = fake
        else:
            prod = '_' if prod else '-'
            num = 1+(cache[prod] or 0)
            cache[prod] = num
        name = "'EM"+prod+str(num)+"'"
        if curly:
            name = '{'+name+'}'
        else:
            name = '<'+name+'>'
        make_temp_embedding(name,vectors,cache,fake)
        return name

    def parse_infotext(text):
        orig = text
        text += '\n'
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
            table = '<style>.webui_embedding_merge_table,.webui_embedding_merge_table td,.webui_embedding_merge_table th{border:1px solid gray;border-collapse:collapse}.webui_embedding_merge_table td,.webui_embedding_merge_table th{padding:2px 5px !important;text-align:center !important;vertical-align:middle;'+font+'font-weight:bold;}.webui_embedding_merge_table{margin:6px auto !important;}</style>'
            (reparse,request) = parse_infotext(gr_text)
            if reparse is not None:
                reparse = parse_mergeseq(reparse)
                if reparse is None:
                    return ('<center><b>Prompt restore failed!</n></center>',gr_name,gr_orig)
                else:
                    request = dict_replace(reparse,request)
                    return ('<center><b>Prompt restored.</n></center>',gr_name,request)
            if gr_text[:1]=="'":
                (two,err) = merge_parser(gr_text,False)
                if (two is not None) and two[0].numel()==0:
                    err = 'Result is ZERO vectors!'
                if err is not None:
                    txt = '<b style="'+font+'">'+html.escape(err)+'</b>'
                else:
                    txt = table
                    both = False
                    for res in two:
                        if res is None:
                            continue
                        if both:
                            txt += '<strong>↑ CLIP (L) / OpenClip (G) ↓</strong>'
                        txt += '<table class="webui_embedding_merge_table"><tr><th>Index</th><th>Min</th><th>Max</th><th>Sum</th><th>Abs</th><th>Len</th><th>Std</th>'
                        i = 1
                        for one in res:
                            txt += '<tr><td>{}</td>{}</tr>'.format(i,tensor_info(one))
                            i += 1
                        txt += '<tr><td colspan="7">&nbsp;</td></tr>'
                        txt += '<tr><td>ALL:</td>{}</tr>'.format(tensor_info(res))
                        txt += '</table>'
                        both = True
                return ('<center>'+txt+'</center>',need_save_embed(store,gr_name,two),gr_orig)
            if gr_text.find("<'")>=0 or gr_text.find("{'")>=0:
                cache = reset_temp_embeddings('-',False)
                used = {}
                (mer,err) = merge_one_prompt(cache,None,{},used,gr_text,False,False)
                if err is not None:
                    txt = '<b style="'+font+'">Embedding Merge failed - '+html.escape(err)+'</b>'
                    return ('<center>'+txt+'</center>',gr_name,gr_orig)
                gr_text = mer
            by_none = 0
            by_comma = 1
            by_parts = 2
            by_words = 3
            by_tokens = 4
            by_vectors = 5
            tok2txt = tokens_to_text()
            if gr_radio!=by_comma:
                two = text_to_vectors(gr_text)
                if (gr_radio==by_none) and (two is not None) and (len(two[0])!=0):
                    two = [[r] for r in two]
            else:
                two = [[],[]]
                split = gr_text.split(',')
                for part in split:
                    one = text_to_vectors(part.strip())
                    if one:
                        two[0].append(one[0])
                        if(len(one)>1):
                            two[1].append(one[1])
                        else:
                            two[1] = None
                    else:
                        two = None
                        break
            if (two is None) or (len(two[0])==0):
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
            both = []
            for res in two:
                if res is None:
                    continue
                txt = '<table class="webui_embedding_merge_table"><tr><th>Index</th><th>Vectors</th><th>Text</th><th>Token</th><th>Min</th><th>Max</th><th>Sum</th><th>Abs</th><th>Len</th><th>Std</th></tr>'
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
                    if gr_radio==by_vectors:
                        head = '<td'+span+'>'+str(size)+'</td>'
                    else:
                        head = '<td'+span+'>'+(str(index) if size==1 else str(index)+'-'+str(index+size-1))+'</td><td'+span+'>'+str(size)+'</td>'
                    if split is None:
                        head += '<td'+span+'>'+html.escape('"'+name+'"')+'</td>'
                    if (gr_radio==by_vectors) or ((gr_radio==by_tokens) and (tokens is not None)):
                        i = 0
                        part = 0
                        j = 0
                        ten = None
                        column = ''
                        toks = None
                        for one in list(tensor):
                            index += 1
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
                            txt += '<tr>{}{}<td>{}</td>{}</tr>'.format(('<td>'+str(index-1)+'</td>' if gr_radio==by_vectors else '')+head,column,tok,tensor_info(use))
                            column = ''
                            head = ''
                            ten = None
                    else:
                        index += size   
                        txt += '<tr>{}<td>{}</td>{}</tr>'.format(head,', '.join([str(t) for t in tokens]) if tokens else '*',tensor_info(tensor))
                txt += '</table>'
                both.append(txt)
            txt = table+'<strong>↑ CLIP (L) / OpenClip (G) ↓</strong>'.join(both)
            return ('<center>'+txt+'</center>',need_save_embed(store,gr_name,two),gr_orig)

    def tensor_info(tensor):
        return '<td>{:>-14.8f}</td><td>{:>+14.8f}</td><td>{:>+14.8f}</td><td>{:>14.8f}</td><td>{:>14.8f}</td><td>{:>14.8f}</td>'.format(tensor.min().item(),tensor.max().item(),tensor.sum().item(),tensor.abs().sum().item(),torch.linalg.norm(tensor,ord=2),tensor.std()).replace(' ','&nbsp;')

    merge_dir = None

    def need_save_embed(store,name,pair):
        if not store:
            return name
        name = ''.join( x for x in name if (x.isalnum() or x in '._- ')).strip()
        if name=='':
            return name
        try:
            if type(pair[0])==list:
                vectors = [torch.cat([r[0] for r in pair[0]])]
                if (len(pair)>1) and (pair[1] is not None):
                    vectors.append(torch.cat([r[0] for r in pair[1]]))
            else:
                vectors = [pair[0]]
                if (len(pair)>1) and (pair[1] is not None):
                    vectors.append(pair[1])
            target = os.path.join(merge_dir,name)
            if len(vectors)>1:
                pt = {
                  'clip_g': vectors[1].cpu(),
                  'clip_l': vectors[0].cpu(),
                }
            else:
                pt = {
                  'string_to_token': {
                    '*': 265,
                  },
                  'string_to_param': {
                    '*': vectors[0].cpu(),
                  },
                  'name': name,
                  'step': 0,
                  'sd_checkpoint': None,
                  'sd_checkpoint_name': None,
                }
            torch.save(pt,target+'.pt')
            try:
                res = torch.load(target+'.pt',map_location='cpu')
            except:
                res = None
            if res is None:
                if len(vectors)==1:
                    pt = {
                      'emb_params': vectors[0].cpu(),
                    }
                from safetensors.torch import save_file
                save_file(pt,target+'.safetensors')
                os.unlink(target+'.pt')
            try:
                modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
            except:
                modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
            return ''
        except:
            traceback.print_exc()
            return name

    def embedding_merge_dir():
        try:
            nonlocal merge_dir
            merge_dir = os.path.join(cmd_opts.embeddings_dir,'embedding_merge')
            # don't actually need this, since it is a subfolder which will be read recursively:
            #modules.sd_hijack.model_hijack.embedding_db.add_embedding_dir(merge_dir)
            os.makedirs(merge_dir)
        except:
            pass

    def raise_sd_error(p,msg):
        class Exception_From_EmbeddingMergeExtension_():
            def __getattribute__(self,_):
                raise Exception_From_EmbeddingMergeExtension(msg)
        p.__class__ = Exception_From_EmbeddingMergeExtension_

    em_regexp = re.compile(r"<'EM[_/-]\d+'>|{'EM[_/-]\d+'}")

    def merge_one_prompt(cache,texts,parts,used,prompt,prod,only_count):
        #if len(get_model_clips())>1:
        #    return (None,'To enable SDXL support switch to "sdxl" branch of https://github.com/klimaleksus/stable-diffusion-webui-embedding-merge')
        try:
            cnt = 0
            if (prompt is None) or (prompt==''):
                return (prompt,None)
            if texts is not None:
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
                    if texts is not None:
                        texts[orig] = prompt
                    return (prompt,None)
                eph = em_regexp.match(prompt[left:])
                if eph is not None:
                    left += len(eph.group(0))
                    continue
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
                    (res,err) = merge_parser(part,only_count)
                    if err is not None:
                        return (None,err)
                    if only_count:
                        if (res is None) or (res==0):
                            embed = ''
                        else:
                            embed = add_temp_embedding(None,cache,prod,curly,res)
                    else:
                        if (res is None) or (res[0].numel()==0):
                            embed = ''
                        else:
                            embed = add_temp_embedding(res,cache,prod,curly,0)
                    if used is not None:
                        used[embed] = part
                    parts[part] = embed
                prefix = prompt[:left].rstrip()+' '+embed
                left = len(prefix)
                prompt = prefix+' '+(prompt[right+1:].lstrip())
        except:
            traceback.print_exc()
            return (None,'Fatal error?')

    fake_cached_params_counter = time.time()
    def fake_cached_params(self,*ar,**kw):
        nonlocal fake_cached_params_counter
        fake_cached_params_counter += 1
        return (*(self.em_orig_cached_params(*ar,**kw)),id(_webui_embedding_merge_),fake_cached_params_counter)

    cached_state = None

    '''
    import hunter
    @hunter.wrap(local=True,actions=[hunter.VarsSnooper,hunter.CallPrinter])
    def pretty_print(clas, indent=0, dupl=None):
        if dupl is None:
            dupl = {}
        me = id(clas)
        tab = ' ' * indent
        if clas is None:
            print(tab + ': None')
            return
        print(tab +  type(clas).__name__ +  ':')
        indent += 4
        tab = ' ' * indent
        if me in dupl:
            print(tab + '[CIRCULAR]')
            return
        dupl[me] = True
        for k,v in clas.__dict__.items():
            if '__dict__' in dir(v):
                pretty_print(v,indent,dupl)
            else:
                print(tab +  k + ': ' + str(v))
    '''

    def hook_infotext(hook):
        if hasattr(processing,'create_infotext'):
            field = '__embedding_merge_wrapper'
            old = getattr(processing,'create_infotext')
            if hasattr(old,field):
                old = getattr(old,field)
                if not hook:
                    setattr(processing,'create_infotext',old)
            if hook:
                def create_infotext(p,*ar,**kw):
                    res = old(p,*ar,**kw)
                    if 'EmbeddingMerge' in p.extra_generation_params:
                        (reparse,request) = parse_infotext(res)
                        if reparse is not None:
                            parse = parse_mergeseq(reparse)
                            matches = em_regexp.findall(request)
                            if (matches is not None) and len(matches)>0:
                                used = {}
                                for match in matches:
                                    used[match] = True
                                gen = ''
                                drop = False
                                for embed,text in parse.items():
                                    if embed in used:
                                        gen += embed+'='+text+', '
                                    else:
                                        drop = True
                                if gen!='' and drop:
                                    gen = gen[:-2]
                                    orig = p.extra_generation_params['EmbeddingMerge']
                                    if gen!=orig:
                                        p.extra_generation_params['EmbeddingMerge'] = gen
                                        res = old(p,*ar,**kw)
                                        p.extra_generation_params['EmbeddingMerge'] = orig
                    return res
                setattr(create_infotext,field,old)
                setattr(processing,'create_infotext',create_infotext)

    def embedding_merge_extension(p,processed):
        if processed is not None:
            hook_infotext(False)
            return
        hook_infotext(True)
        nonlocal cached_state
        use_hr = hasattr(p,'hr_prompt')
        arr = [
            p.all_prompts,
            p.prompt if type(p.prompt)==list else [p.prompt],
            p.all_negative_prompts,
            p.negative_prompt if type(p.negative_prompt)==list else [p.negative_prompt],
        ]
        if use_hr:
            arr += [
                p.all_hr_prompts,
                p.hr_prompt if type(p.hr_prompt)==list else [p.hr_prompt],
                p.all_hr_negative_prompts,
                p.hr_negative_prompt if type(p.hr_negative_prompt)==list else [p.hr_negative_prompt],
            ]
        restart = True
        if 'EmbeddingMerge' in p.extra_generation_params:
            restart = False
        elif em_regexp.search(' '.join([' '.join(one) for one in arr if one is not None])) is not None:
            restart = False
            print("[EmbeddingMerge] WARNING: ephemeral embeddings (like <'EM_1'>) are detected!")
        if restart or (cached_state is None):
            cached_state = {
                'cache': reset_temp_embeddings('_',False),
                'texts': {},
                'parts': {},
                'used': {},
            }
        cache = cached_state['cache']
        texts = cached_state['texts']
        parts = cached_state['parts']
        used = cached_state['used']
        for one in arr:
            ok = False
            fail = None
            if one is not None:
                for i in range(len(one)):
                    (res,err) = merge_one_prompt(cache,texts,parts,used,one[i],True,False)
                    if err is not None:
                        if fail is None:
                            fail = err
                    else:
                        one[i] = res
                        ok = True
            if not ok and fail is not None:
                raise_sd_error(p,'\n\nEmbedding Merge failed - '+err+'\n')
                return
        p.all_prompts = arr[0]
        p.all_negative_prompts = arr[2]
        p.prompt = arr[1] if type(p.prompt)==list else arr[1][0]
        p.negative_prompt = arr[3] if type(p.negative_prompt)==list else arr[3][0]
        if use_hr:
            p.all_hr_prompts = arr[4]
            p.all_hr_negative_prompts = arr[6]
            p.hr_prompt = arr[5] if type(p.hr_prompt)==list else arr[5][0]
            p.hr_negative_prompt = arr[7] if type(p.hr_negative_prompt)==list else arr[7][0]
        gen = ''
        was_used = False
        for embed in used:
            was_used = True
            if embed!='':
                if embed[0]=='<':
                    gen += embed+'=<'+used[embed]+'>, '
                else:
                    gen += embed+'={'+used[embed]+'}, '
        if gen!='':
            p.extra_generation_params['EmbeddingMerge'] = gen[:-2]
        if was_used:
            orig = getattr(p,'cached_params',None)
            if orig is not None:
                setattr(p,'em_orig_cached_params',orig)
                setattr(p,'cached_params',types.MethodType(fake_cached_params,p))

    try:
        cls = modules.sd_hijack.StableDiffusionModelHijack
        get_prompt_lengths = cls.get_prompt_lengths
        field = '__embedding_merge_wrapper'
        def hook_prompt_lengths(self,text,*ar,**kw):
            if text.find("<'")<0 and text.find("{'")<0:
                return get_prompt_lengths(self,text,*ar,**kw)
            (res,err) = merge_one_prompt(grab_embedding_cache(),None,{},None,text,True,True)
            if err is not None:
                return -1,-1
            return get_prompt_lengths(self,res,*ar,**kw)
        if hasattr(get_prompt_lengths,field):
            get_prompt_lengths = getattr(get_prompt_lengths,field)
        setattr(hook_prompt_lengths,field,get_prompt_lengths)
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
            if 'Hires prompt' in result:
                result['Hires prompt'] = dict_replace(reparse,result['Hires prompt'])
            if 'Hires negative prompt' in result:
                result['Hires negative prompt'] = dict_replace(reparse,result['Hires negative prompt'])
    setattr(_webui_embedding_merge_,'on_infotext_pasted',on_infotext_pasted)
    def on_model_loaded(*ar,**kw):
        reset_temp_embeddings('/',True)
    setattr(_webui_embedding_merge_,'on_model_loaded',on_model_loaded)

    def on_script_unloaded():
        hook_infotext(False)
        reset_temp_embeddings('_',True)
        reset_temp_embeddings('-',True)
        reset_temp_embeddings('/',True)
        try:
            cls = modules.sd_hijack.StableDiffusionModelHijack
            get_prompt_lengths = cls.get_prompt_lengths
            field = '__embedding_merge_wrapper'
            if hasattr(get_prompt_lengths,field):
                cls.get_prompt_lengths = getattr(get_prompt_lengths,field)
        except:
            traceback.print_exc()
        try:
            db = modules.sd_hijack.model_hijack.embedding_db
            field = '__embedding_merge_cache_'
            if hasattr(db,field):
                delattr(db,field)
        except:
            traceback.print_exc()
    setattr(_webui_embedding_merge_,'on_script_unloaded',on_script_unloaded)
    setattr(_webui_embedding_merge_,'embedding_merge_extension',embedding_merge_extension)
    embedding_merge_dir()
    return gr_tab

class EmbeddingMergeExtension(scripts.Script):
    def title(self):
        return 'Embedding Merge'
    def show(self,is_img2img):
        return scripts.AlwaysVisible
    def process(self,p):
        if hasattr(_webui_embedding_merge_,'embedding_merge_extension'):
            getattr(_webui_embedding_merge_,'embedding_merge_extension')(p,None)
    def postprocess(self,p,processed):
        if hasattr(_webui_embedding_merge_,'embedding_merge_extension'):
            getattr(_webui_embedding_merge_,'embedding_merge_extension')(p,processed)

script_callbacks.on_ui_tabs(_webui_embedding_merge_())
script_callbacks.on_infotext_pasted(_webui_embedding_merge_.on_infotext_pasted)
script_callbacks.on_script_unloaded(_webui_embedding_merge_.on_script_unloaded)
try:
    script_callbacks.on_model_loaded(_webui_embedding_merge_.on_model_loaded)
except:
    pass

#EOF
