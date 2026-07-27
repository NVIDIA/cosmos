from ..smp import *
import os
import sys
import mimetypes
from .base import BaseAPI


API_BASE = "http://localhost:8009"


class LVSWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'lvs',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 chunk_duration: int = 10,
                 num_frames_per_chunk: int = 10,
                 temperature: float = 0.1,
                 api_base: str = None,
                 max_tokens: int = 16384,
                 fail_msg: dict = {"events": [], "total_events": 0, "video_summary": ""},
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_base = api_base if api_base is not None else API_BASE
        self.chunk_duration = chunk_duration
        self.num_frames_per_chunk = num_frames_per_chunk

        self.key = key
        self.fail_msg = fail_msg
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if api_base is None:
            self.logger.info(f'API Base not provided. Using default API Base: {API_BASE}')
        else:
            self.logger.info(f'Using API Base: {self.api_base}; API Key: {self.key}')

        import requests
        try:
            response = requests.get(
                self.api_base + '/v1/live',
            )
            ret_code = response.status_code
            if 200 <= int(ret_code) < 300:
                self.logger.info(
                    f'Successfully connected to LVS at {self.api_base}. '
                    f'Health check returned status code: {ret_code}'
                    f'Ensure LVS is deployed with /files API enabled by setting VIA_DEV_API=true'
                )
            else:
                raise ConnectionError(
                    f'LVS is not available at {self.api_base}. '
                    f'Health check failed with status code: {ret_code}'
                )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f'LVS is not available at {self.api_base}. '
                f'Connection error: {e}'
            )

    def generate_inner(self, inputs, *args, **kwargs) -> str:
        video_url = None
        text_prompt = None
        
        for item in inputs:
            if item['type'] == 'video':
                video_url = item['value']
            elif item['type'] == 'text':
                text_prompt = item['value']
        
        if video_url is None:
            raise ValueError('No video URL found in inputs')
        if text_prompt is None:
            raise ValueError('No text prompt found in inputs')
        
        self.logger.info(f'Video path: {video_url}')
        self.logger.info(f'Text Prompt: {text_prompt[:50] if len(text_prompt) > 50 else text_prompt}')

        # Upload file to LVS using multipart form-data
        # Equivalent to: curl --form 'file=@"/path/to/video.mp4"' --form 'purpose="vision"' --form 'media_type="video"'
        
        # Verify file exists
        if not os.path.exists(video_url):
            log = f'Video file not found: {video_url}'
            self.logger.error(log)
            return 404, json.dumps(self.fail_msg), log
        
        # Get filename and determine MIME type
        filename = os.path.basename(video_url)
        mime_type, _ = mimetypes.guess_type(video_url)
        if mime_type is None:
            mime_type = 'video/mp4'  # Default to mp4
        
        self.logger.info(f'Uploading file: {filename} (MIME: {mime_type})')
        
        try:
            with open(video_url, 'rb') as video_file:
                # Build multipart form-data request
                files = {
                    'file': (filename, video_file, mime_type)
                }
                data = {
                    'purpose': 'vision',
                    'media_type': 'video'
                }
                
                upload_response = requests.post(
                    f'{self.api_base}/files',
                    files=files,
                    data=data
                )
        except IOError as e:
            log = f'Failed to read video file: {video_url}, error: {e}'
            self.logger.error(log)
            return 500, json.dumps(self.fail_msg), log
        
        ret_code = upload_response.status_code
        if not (200 <= ret_code < 300):
            log = f'Failed to upload file. Status: {ret_code}, Response: {upload_response.text}'
            self.logger.error(log)
            return ret_code, json.dumps(self.fail_msg), log
        
        # Get file_id from upload response
        try:
            upload_data = upload_response.json()
            file_id = upload_data['id']
        except (json.JSONDecodeError, KeyError) as e:
            log = f'Failed to parse upload response: {upload_response.text}, error: {e}'
            self.logger.error(log)
            return ret_code, json.dumps(self.fail_msg), log
        
        self.logger.info(f'Successfully uploaded file. file_id: {file_id}')

        # Build summarize request
        summarize_data = {
            'id': [file_id],
            'model': self.model,
            'auto_generate_prompt': True,
            'chunk_duration': self.chunk_duration,
            'num_frames_per_chunk': self.num_frames_per_chunk,
        }

        try:
            prompt_data = json.loads(text_prompt)
            if isinstance(prompt_data, dict):
                events = prompt_data.get('event_list', prompt_data.get('events', []))
                scenario = prompt_data.get('scenario', '')
                # If events is a string (pipe-separated), convert to list
                if isinstance(events, str):
                    events = [e.strip() for e in events.split('|')]
                summarize_data['events'] = events
                summarize_data['scenario'] = scenario
                self.logger.info(f'Sending events: {events}, scenario: {scenario}')
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, use text_prompt as-is as prompt
            summarize_data['prompt'] = text_prompt

        # Call summarize API
        self.logger.info('Calling /summarize API...')
        summarize_response = requests.post(
            f'{self.api_base}/summarize',
            headers={'Content-Type': 'application/json'},
            json=summarize_data
        )

        http_status = summarize_response.status_code
        if 200 <= http_status < 300:
            try:
                response_data = summarize_response.json()
            except json.JSONDecodeError:
                log = f'Failed to parse summarize response as JSON: {summarize_response.text}'
                self.logger.error(log)
                # Return non-zero ret_code to trigger retry
                return 1, json.dumps(self.fail_msg), log

            try:
                choices = response_data.get('choices', [])
                if choices and len(choices) > 0:
                    message = choices[0].get('message', {})
                    content = message.get('content', '')

                    # Parse the content JSON
                    if content:
                        content_dict = json.loads(content)
                        events = content_dict.get('events', [])

                        answer = json.dumps(events)
                        log = f'Successfully called summarize. Found {len(events)} events.'
                        self.logger.info(log)
                        # Return 0 for success (BaseAPI expects ret_code == 0 for success)
                        return 0, answer, log
                    else:
                        log = f'Empty content in response: {response_data}'
                        self.logger.warning(log)
                        return 1, json.dumps(self.fail_msg), log
                else:
                    log = f'No choices found in response: {response_data}'
                    self.logger.error(log)
                    return 1, json.dumps(self.fail_msg), log

            except Exception as e:
                log = f'Error parsing response: {e}, response_data: {response_data}'
                self.logger.error(log)
                return 1, json.dumps(self.fail_msg), log
        else:
            log = f'Failed to summarize. Status: {http_status}, Response: {summarize_response.text}'
            self.logger.error(log)
            # Return HTTP status code (non-zero) to trigger retry
            return http_status, json.dumps(self.fail_msg), log


class LVS_service(LVSWrapper):
    VIDEO_LLM = True

    def generate(self, message, dataset=None):
        return super(LVS_service, self).generate(message)
