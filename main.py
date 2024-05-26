import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage

from HR import SalesGPT, llm
#### !! ADD bot_token !! ####
bot_token = ''
sales_agent = None

async def main():

    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    bot = Bot(bot_token, parse_mode=None)
    logging.basicConfig(level=logging.INFO)
    
    @dp.message(Command(commands=["start"]))
    async def repl(message):
        global sales_agent
        sales_agent = SalesGPT.from_llm(llm, verbose=False)
        sales_agent.seed_agent()
        ai_message = sales_agent.ai_step()
        await message.answer(ai_message)
    
    @dp.message(F.text)
    async def repl(message):
        if sales_agent is None:
            await message.answer('Используйте команду /start')
        else:
            human_message = message.text
            if human_message:
                sales_agent.human_step(human_message)
                sales_agent.analyse_stage()
            ai_message = sales_agent.ai_step()
            await message.answer(ai_message)

    @dp.message(~F.text)
    async def empty(message):
        await message.answer('Бот принимает только текст')

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())