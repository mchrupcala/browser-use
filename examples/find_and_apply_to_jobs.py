"""
Find and apply to jobs.

@dev You need to add OPENAI_API_KEY to your environment variables.

Also you have to install PyPDF2 to read pdf files: pip install PyPDF2
"""

import csv
import os
import re
import sys
from pathlib import Path

from PyPDF2 import PdfReader

from browser_use.browser.browser import Browser, BrowserConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, SecretStr

from browser_use import ActionResult, Agent, Controller
from browser_use.browser.context import BrowserContext

load_dotenv()
import logging

logger = logging.getLogger(__name__)
# full screen mode
controller = Controller()
CV = Path.cwd() / 'cv_04_24.pdf'


class Job(BaseModel):
	title: str
	link: str
	company: str
	fit_score: float
	location: Optional[str] = None
	salary: Optional[str] = None


@controller.action(
	'Save jobs to file - with a score how well it fits to my profile', param_model=Job
)
def save_jobs(job: Job):
	with open('jobs.csv', 'a', newline='') as f:
		writer = csv.writer(f)
		writer.writerow([job.title, job.company, job.link, job.salary, job.location])

	return 'Saved job to file'


@controller.action('Read jobs from file')
def read_jobs():
	with open('jobs.csv', 'r') as f:
		return f.read()


@controller.action('Read my cv for context to fill forms')
def read_cv():
	pdf = PdfReader(CV)
	text = ''
	for page in pdf.pages:
		text += page.extract_text() or ''
	logger.info(f'Read cv with {len(text)} characters')
	return ActionResult(extracted_content=text, include_in_memory=True)


@controller.action(
	'Upload cv to element - call this function to upload if element is not found, try with different index of the same upload element',
	requires_browser=True,
)
async def upload_cv(index: int, browser: BrowserContext):
	path = str(CV.absolute())
	dom_el = await browser.get_dom_element_by_index(index)

	if dom_el is None:
		return ActionResult(error=f'No element found at index {index}')

	file_upload_dom_el = dom_el.get_file_upload_element()

	if file_upload_dom_el is None:
		logger.info(f'No file upload element found at index {index}')
		return ActionResult(error=f'No file upload element found at index {index}')

	file_upload_el = await browser.get_locate_element(file_upload_dom_el)

	if file_upload_el is None:
		logger.info(f'No file upload element found at index {index}')
		return ActionResult(error=f'No file upload element found at index {index}')

	try:
		await file_upload_el.set_input_files(path)
		msg = f'Successfully uploaded file to index {index}'
		logger.info(msg)
		return ActionResult(extracted_content=msg)
	except Exception as e:
		logger.debug(f'Error in set_input_files: {str(e)}')
		return ActionResult(error=f'Failed to upload file to index {index}')


browser = Browser(
    config=BrowserConfig(
        headless=False,
        disable_security=True,
    )
)


async def main():
	# ground_task = (
	# 	'You are a professional job finder. '
	# 	'1. Read my cv with read_cv'
	# 	'2. Read the saved jobs file '
	# 	'3. start applying to the first link of Amazon '
	# 	'You can navigate through pages e.g. by scrolling '
	# 	'Make sure to be on the english version of the page'
	# )
	ground_task = (
		'You are a professional job finder. '
		'1. Read my cv with read_cv'
		'Use my LinkedIn credentials to log in and find and apply to software engineer jobs.'
		'The only jobs you should apply for should be mid-level roles (2 to 4 years of experience required) AND the job title should include the words "frontend" or "fullstack" or "front end" or "full stack" or "front-end" or "full-stack.'
		'You should NOT apply for hybrid or on-site or in-person or office-only roles unless they are based in New York City or Brooklyn.'
		'If you need to upload a cv or resume and you need to find the correct upload button, look for a UI element with the button text "Upload file"'
    f' The username is michael.chrupcala@gmail.com and the password is: {os.getenv("LINKEDIN_PASSWORD")}'
	'After you finish filling out an application and you reach a page with the text "Review your application", you will need to click a button with the words "Submit application" to finish submitting this application.'
	'If you reach the "review application" stage and you reach a screen or modal with "Save this application?" then click "Save" or "Yes" and continue.' 
	'After you successfully apply for a job, search for another and continue to submit applications.'
	)
	tasks = [
		ground_task + '\n' + 'LinkedIn',
		# ground_task + '\n' + 'Amazon',
		# ground_task + '\n' + 'Apple',
		# ground_task + '\n' + 'Microsoft',
		# ground_task + '\n' + 'Meta',
	]
	model = ChatOpenAI(
		model='gpt-4o',
		# api_version='2024-10-21',
		# azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
		# api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
	)

	agents = []
	for task in tasks:
		agent = Agent(task=task, llm=model, controller=controller, browser=browser)
		agents.append(agent)

	await asyncio.gather(*[agent.run() for agent in agents])


if __name__ == '__main__':
	asyncio.run(main())
