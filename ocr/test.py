from typing import List

def merge_continuous_sentences(lines: List[str]) -> List[str]:
    """
    Merge lines if:
    1. Current line does NOT end with punctuation (.!?:;)
    2. Next line starts with a lowercase letter
    3. OR Current line has only one word (when stripped)
    4. OR Last word starts with uppercase AND next line starts with uppercase
    5. OR Current line doesn't end with punctuation AND next line starts with uppercase
    """
    if not lines:
        return lines
    
    merged = []
    i = 0
    
    while i < len(lines):
        original_line = lines[i]
        stripped_line = original_line.strip()
        
        if not stripped_line:
            merged.append(original_line)
            i += 1
            continue
        
        merged_line = original_line
        
        while i + 1 < len(lines):
            next_original = lines[i + 1]
            next_stripped = next_original.strip()
            
            if not next_stripped:
                break
            
            # Get current state of the merged line
            current_stripped = merged_line.strip()
            
            # Check if current line ends with punctuation
            ends_with_punct = any(current_stripped.endswith(p) for p in {'.', '!', '?', ':', ';', ')', ']', '}', '"', "'"})
            
            # Check if current line has only one word
            has_one_word = len(current_stripped.split()) == 1
            
            # Check if last word starts with uppercase
            current_words = current_stripped.split()
            last_word_upper = False
            if current_words:
                last_word = current_words[-1]
                # Remove any trailing punctuation or special characters to check the word itself
                clean_word = ''.join(c for c in last_word if c.isalpha())
                if clean_word and clean_word[0].isupper():
                    last_word_upper = True
            
            # Check if next line's first word starts with uppercase
            next_words = next_stripped.split()
            next_starts_upper = False
            next_starts_lower = False
            if next_words:
                first_word = next_words[0]
                # Remove any leading punctuation or special characters to check the word itself
                clean_word = ''.join(c for c in first_word if c.isalpha())
                if clean_word:
                    if clean_word[0].isupper():
                        next_starts_upper = True
                    elif clean_word[0].islower():
                        next_starts_lower = True
            
            # Determine if we should merge - CHECK CONDITIONS IN PRIORITY ORDER
            should_merge = False
            
            # Priority 1: One word lines (always merge)
            if has_one_word:
                should_merge = True
            # Priority 2: Uppercase to uppercase (specific case you want)
            elif last_word_upper and next_starts_upper:
                should_merge = True
            # Priority 3: No punctuation + lowercase (standard sentence continuation)
            elif not ends_with_punct and next_starts_lower:
                should_merge = True
            # Priority 4: No punctuation + uppercase (general case)
            elif not ends_with_punct and next_starts_upper:
                should_merge = True
            
            if should_merge:
                # Merge with the next line's content
                merged_line = merged_line.rstrip() + ' ' + next_stripped
                i += 1
            else:
                break
        
        merged.append(merged_line)
        i += 1
    
    return merged

# Test cases
test_cases = [
    {
        "input": [
            " Y ednachVorlagedecBeratangsprota.cllPrafungshommissionen im (1saenz.Masterokra",
            " Universitijahres2015-2016."
        ],
        "description": "Complex academic text with uppercase continuation"
    },
    {
        "input": [
            "This is a sentence",
            "that continues here"
        ],
        "description": "Standard sentence continuation"
    },
    {
        "input": [
            "SingleWord",
            "should merge"
        ],
        "description": "Single word line merging"
    },
    {
        "input": [
            "Title Case",
            "Subtitle Here"
        ],
        "description": "Uppercase to uppercase merging"
    },
    {
        "input": [
            "Sentence ending.",
            "New sentence"
        ],
        "description": "Should NOT merge (ends with punctuation)"
    },
    {
        "input": [
            "Line with indentation   ",
            "    should preserve spaces"
        ],
        "description": "Indentation preservation"
    }
]

print("Running test cases:\n")
for case in test_cases:
    print(f"Test: {case['description']}")
    print("Input lines:")
    for line in case["input"]:
        print(f"  '{line}'")
    
    result = merge_continuous_sentences(case["input"])
    
    print("\nResult:")
    for line in result:
        print(f"  '{line}'")
    print("\n" + "-"*50 + "\n")