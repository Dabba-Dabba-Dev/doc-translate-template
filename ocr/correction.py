def correct_text(self, text):
        response = requests.post(
            "https://api.languagetool.org/v2/check",
            data={
                "text": text,
                "language": "auto"
            }
        )
        suggested_text = text
        offset_correction = 0

        for match in response.json().get("matches", []):
            if match['replacements']:
                start = match['offset'] + offset_correction
                end = start + match['length']
                replacement = match['replacements'][0]['value']
                suggested_text = suggested_text[:start] + replacement + suggested_text[end:]
                offset_correction += len(replacement) - match['length']

        return suggested_text.strip()
def process_image(self, image_path):
        raw_text = self.extract_text(image_path)
        corrected_text = self.correct_text(raw_text)
        return raw_text, corrected_text
corrected=ocr.process_image('ffff_page-0001.jpg')
print("\n✅ Corrected Text:\n", corrected)