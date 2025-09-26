import type { ThemeNoteListOptions } from 'vuepress-theme-plume'
import { defineNotesConfig } from 'vuepress-theme-plume'
// import { plugins } from './plugins'
// import { themeConfig } from './theme-config'
import { Guide } from './guide'

export const zhNotes: ThemeNoteListOptions = defineNotesConfig({
    dir: 'zh/notes',
    link: '/zh/',
    notes: [
        Guide,
    ],
})
