import type { ThemeNoteListOptions } from 'vuepress-theme-plume'
import { defineNotesConfig } from 'vuepress-theme-plume'
import { Guide } from './guide'

export const enNotes: ThemeNoteListOptions = defineNotesConfig({
    dir: 'en/notes',
    link: '/en/',
    notes: [
        Guide,
    ],
})
