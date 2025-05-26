import json
from datetime import datetime

def process_chat_log(file_path):
    """Process Telegram chat log JSON for PayWay integration analysis"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for msg in data.get('messages', []):
        # Extract base message info
        entry = {
            'date': datetime.fromisoformat(msg.get('date', '')).strftime('%Y-%m-%d %H:%M:%S') if msg.get('date') else 'Unknown date',
            'sender': msg.get('from', ''),
            'message_type': msg.get('type', ''),
            'text': '',
            'files': [],
            'reactions': []
        }

        # Process text content
        text_entities = []
        for entity in msg.get('text_entities', []):
            if entity['type'] == 'plain':
                text_entities.append(entity['text'])
            elif entity['type'] in ['mention', 'link', 'email', 'phone']:
                text_entities.append(f"[{entity['type']}: {entity['text']}]")
        entry['text'] = ' '.join(text_entities)

        # Extract files
        if 'file' in msg:
            entry['files'].append({
                'name': msg.get('file_name', ''),
                'type': msg.get('mime_type', ''),
                'size': msg.get('file_size', 0)
            })

        # Extract reactions
        for reaction in msg.get('reactions', []):
            entry['reactions'].append({
                'emoji': reaction['emoji'],
                'count': reaction['count']
            })

        results.append(entry)
    
    # Generate report
    report = {
        'total_messages': len(results),
        'participants': list({msg['sender'] for msg in results}),
        'timeline': {
            'first_message': results[0]['date'] if results else None,
            'last_message': results[-1]['date'] if results else None
        },
        'messages_by_type': {}
    }
    
    return results, report

# Example usage
if __name__ == "__main__":
    try:
        messages, analysis = process_chat_log('data/raw_exports/Viwat  Default Payway integration.json')
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        exit(1)
    
    print(f"Analysis Report:")
    print(f"Total messages: {analysis['total_messages']}")
    print(f"Participants: {', '.join(analysis['participants'])}")
    print(f"Timeline: {analysis['timeline']['first_message']} to {analysis['timeline']['last_message']}")
    
    print("\nSample Messages:")
    for msg in messages[:3]:
        print(f"\n[{msg['date']}] {msg['sender']}: {msg['text']}")
        if msg['files']:
            print(f"Attachments: {len(msg['files'])} files")
