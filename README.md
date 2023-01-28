_This is WIP, the following helpfile is taken from the extension itself._

# webui-embedding-merge

This is extension for https://github.com/AUTOMATIC1111/stable-diffusion-webui  
It creates a tab named "EM"

## View text or embeddings vectors

You can paste your vanilla prompt (without any other special syntax) into the textbox below to see how it is parsed by WebUI. All of detected Textual Inversion embeddings will be extracted and presented to you along with literal text tokens. For example:

>intergalactic train, masterpiece, by Danh Víµ

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

## Test merge expression:

You can enter a "merge expression" that starts with a single quote, to see how it will be parsed and combined by this extension. It should contain single quotes around literal texts or TI embeggings, and special operators between them. For example:

>'greg rutkowski'/4+'gustav dore'*0.75

### expression syntax...

- `'one' + 'two'` = blend vectors together by simple sum of all values. If length is different, smallest part will be right-padded with zeroes.
- `'one' - 'two'` = as above, but subtraction. Note that + and - can be put only between textual parts and will have lowest priority.
- `'text' * NUM` = multiply all vectors of quoted literal by numeric value. You can use floating point (0.85) and negative numbers (-1), but not arithmetic expressions.
- `'text' / NUM` = division by number, just as multiplication above. Applies to previous text literal but after previous similar operations, so you can multiply and divide together (\*3/5)
- `'text' : NUM` = change vector count of literal, to shrink or enlarge (padded with zeros). Only integer without sign!
- `'text' :+ NUM` and `'text'  :- NUM` = circular rotate vectors in this token, for example +1 will shift index of each vector by one forward, wrapping on last.

To apply multiplication (or division), cropping or shifting *to the result* of addition (or subtraction), you cannot use parenthesis; instead, try this syntax:

- `'one' + 'two' =* NUM` = will multiply the sum of 'one' and 'two', but not 'two' alone
- `'one' + 'two' =/ NUM` = divide the sum (or any number of sums to the left), effectively the "result" of everything
- `'one' + 'two' =: NUM` = crop or enlarge the results
- `'one' + 'two' =:+ NUM` or `'one' + 'two' =:- NUM` = rotate the result

Thus, the following operations are doing the same:

>`'a'/2 + 'b'/2 + '':1 - 'd'`   
`'a'+'b' =* 0.5 + 'c'*0 + 'd'*-1`

## Several merge expressions in prompt:

If you put a valid merge expression enclosed in angular <'…' …> or curly {'…' …} brackets anywhere in your prompt (with no space between `<` or `{` and `'`), it will be parsed and merged into one temporary Textual Inversion embedding, which replaces the expression itself. The resulting prompt will be joined from those embeddings and anything between expressions. For example:

>A photo of <'cat'+'dog'>, {'4k'+'dynamic lighting'+'science fiction'=/3} masterpiece

### More examples

TODO

## Using merge expressions in prompts at runtime!

You can actually put merge expressions in angular or curly brackets into your txt2img or img2img prompt in WebUI. This extension will intercept both main and negative prompts, parse and merge expressions creating temporary TI embeddings that WebUI will "see" instead of your original text. In generation info there will be internal meaningless names like <'EM_1'>, but extra parameter "EmbeddingMerge" will contain original merge expressions. To quickly restore your prompts, just paste your complete generation information (from .txt or PNG Info) into the textbox on this tab (also it should work for the official "paste" toolbar button too) – its temporary embeddings will be replaced back with expressions, for example:

> a photo of <'EM_1'>  
Negative prompt: {'EM_2'}  
Steps: 8, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1374372309, Size: 512x512, Model hash: c6bbc15e32, Model: sd-v1-5-inpainting, EmbeddingMerge: "<'EM_1'>=<'sky' * 2/4 + 'forest' * 3/4>, {'EM_2'}={'blurry'+'cropped'}", Conditional mask weight: 1

### Technical information...

TODO
