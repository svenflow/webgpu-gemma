import type { ConversationTurn } from './types.js';

/**
 * Build a chat prompt string from conversation history with optional tool declarations.
 */
export function buildChatPrompt(history: ConversationTurn[], toolsJson?: string): string {
  let systemPrefix = '';
  let tools: any[] | null = null;
  try {
    if (toolsJson) {
      const parsed = JSON.parse(toolsJson);
      if (Array.isArray(parsed) && parsed.length > 0) tools = parsed;
    }
  } catch {
    tools = null;
  }

  if (tools) {
    let declarations = '';
    for (const tool of tools) {
      declarations += `<start_function_declaration>declaration:${tool.name}{`;
      declarations += `description:<escape>${tool.description}<escape>`;
      if (tool.parameters) {
        declarations += `,parameters:{properties:{`;
        const props = Object.entries(tool.parameters.properties || {});
        declarations += props.map(([k, v]: [string, any]) => {
          let param = `${k}:{description:<escape>${v.description}<escape>,type:<escape>${v.type}<escape>`;
          if (v.enum) {
            param += `,enum:[${v.enum.map((e: string) => `<escape>${e}<escape>`).join(',')}]`;
          }
          param += '}';
          return param;
        }).join(',');
        declarations += `}`;
        if (tool.parameters.required) {
          declarations += `,required:[${tool.parameters.required.map((r: string) => `<escape>${r}<escape>`).join(',')}]`;
        }
        declarations += `,type:<escape>${tool.parameters.type}<escape>`;
        declarations += `}`;
      }
      declarations += `}<end_function_declaration>`;
    }
    systemPrefix = `<start_of_turn>developer\nYou are a model that can do function calling with the following functions\n${declarations}\n<end_of_turn>\n`;
  }

  let prompt = systemPrefix;
  for (const turn of history) {
    prompt += `<start_of_turn>${turn.role}\n${turn.text}<end_of_turn>\n`;
  }
  prompt += `<start_of_turn>model\n`;
  return prompt;
}
