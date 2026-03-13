import type { GGUFParsed } from './types.js';

const SPECIAL_TOKENS: Record<string, number> = {
  '<start_of_turn>': 105,
  '<end_of_turn>': 106,
  '<eos>': 1,
  '<bos>': 2,
};

const FUNC_TOKEN_NAMES = [
  '<start_function_declaration>', '<end_function_declaration>',
  '<start_function_call>', '<end_function_call>',
  '<start_function_response>', '<end_function_response>',
  '<escape>',
];

export class Tokenizer {
  vocab: string[] = [];
  vocabByLength: [number, string][] = [];
  tokenByText: Map<string, number> = new Map();
  maxTokenLen: number = 0;
  specialTokens: Record<string, number> = { ...SPECIAL_TOKENS };
  funcTokens: Record<string, number> = {};
  private specialPatternRegex: RegExp = /\\<start_of_turn\\>|\\<end_of_turn\\>|\\<eos\\>|\\<bos\\>/g;

  extractFromGGUF(gguf: GGUFParsed): void {
    const tokensKV = gguf.kv.get('tokenizer.ggml.tokens');
    if (tokensKV && tokensKV.type === 'array') {
      this.vocab = tokensKV.value.map((v: any) => v.value);
    } else {
      throw new Error('No tokenizer found in GGUF metadata');
    }
    this.vocabByLength = [];
    for (let i = 0; i < this.vocab.length; i++) {
      if (this.vocab[i] && this.vocab[i].length > 0) {
        this.vocabByLength.push([i, this.vocab[i]]);
      }
    }
    this.buildTokenIndex();
    this.initFunctionTokens();
  }

  private buildTokenIndex(): void {
    this.tokenByText = new Map();
    this.maxTokenLen = 0;
    for (let i = 0; i < this.vocab.length; i++) {
      const token = this.vocab[i];
      if (token && token.length > 0) {
        if (!this.tokenByText.has(token)) {
          this.tokenByText.set(token, i);
        }
        if (token.length > this.maxTokenLen) {
          this.maxTokenLen = token.length;
        }
      }
    }
  }

  private initFunctionTokens(): void {
    for (const name of FUNC_TOKEN_NAMES) {
      const id = this.tokenByText.get(name);
      if (id !== undefined) {
        this.funcTokens[name] = id;
        this.specialTokens[name] = id;
      }
    }
    this.rebuildSpecialPattern();
  }

  private rebuildSpecialPattern(): void {
    const allTokenNames = Object.keys(this.specialTokens);
    allTokenNames.sort((a, b) => b.length - a.length);
    const specialPatternStr = allTokenNames.map(t => t.replace(/[<>]/g, c => '\\' + c)).join('|');
    this.specialPatternRegex = new RegExp(specialPatternStr, 'g');
  }

  private encodeSegment(text: string, addPrefix: boolean = true): number[] {
    const tokens: number[] = [];
    let remaining = text.replace(/ /g, '\u2581');
    if (addPrefix) {
      remaining = '\u2581' + remaining;
    }
    while (remaining.length > 0) {
      let bestLen = 0;
      let bestId = -1;
      const tryLen = Math.min(remaining.length, this.maxTokenLen);
      for (let len = tryLen; len >= 1; len--) {
        const candidate = remaining.substring(0, len);
        const id = this.tokenByText.get(candidate);
        if (id !== undefined) {
          bestLen = len;
          bestId = id;
          break;
        }
      }
      if (bestLen === 0) {
        remaining = remaining.slice(1);
      } else {
        tokens.push(bestId);
        remaining = remaining.slice(bestLen);
      }
    }
    return tokens;
  }

  encode(text: string): number[] {
    const tokens = [2]; // BOS
    const specialPattern = new RegExp(this.specialPatternRegex.source, 'g');
    let lastIdx = 0;
    let match: RegExpExecArray | null;
    let afterSpecial = false;
    while ((match = specialPattern.exec(text)) !== null) {
      const before = text.slice(lastIdx, match.index);
      if (before.length > 0) {
        tokens.push(...this.encodeSegment(before, !afterSpecial));
      }
      tokens.push(this.specialTokens[match[0]]);
      lastIdx = match.index + match[0].length;
      afterSpecial = true;
    }
    const remaining = text.slice(lastIdx);
    if (remaining.length > 0) {
      tokens.push(...this.encodeSegment(remaining, !afterSpecial));
    }
    return tokens;
  }

  decodeToken(tokenId: number): string {
    if (tokenId < this.vocab.length && this.vocab[tokenId]) {
      return this.vocab[tokenId].replace(/\u2581/g, ' ');
    }
    return `<unk:${tokenId}>`;
  }

  decodeTokens(tokenIds: number[]): string {
    let text = '';
    for (const id of tokenIds) {
      text += this.decodeToken(id);
    }
    return text;
  }
}
