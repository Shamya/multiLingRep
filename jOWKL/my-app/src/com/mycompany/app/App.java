package com.mycompany.app;

/**
 * Hello world!
 *
 */
import java.io.*;
import java.net.*;

import org.dkpro.jowkl.api.*;
import org.dkpro.jowkl.exception.*;
import org.dkpro.jowkl.db.*;
import java.util.*;


class App {
    public static void main(String args[]) throws Exception {
        String ow_host = "localhost";
        String ow_db = "owl";
        String ow_user = "newuser";
        String ow_pass = "password";
        String db_driver = "com.mysql.jdbc.Driver"; //just an example, other drivers should work too
        String db_vendor = "mysql";
        int ow_language= OWLanguage.English;
        DatabaseConfiguration dbConfig_ow = new DatabaseConfiguration(ow_host,ow_db,db_driver,db_vendor, ow_user, ow_pass, ow_language);
        //Create the OmegaWiki object
        OmegaWiki ow = new OmegaWiki(dbConfig_ow);
        //Retrieve all senses for the English word "table
        Set<DefinedMeaning> meanings = ow.getDefinedMeaningByWord("table", ow_language);
        //For all senses...
        for(DefinedMeaning dm : meanings)
        {
            //Retrieve the English definitions
            Set<TranslatedContent> glosses = dm.getGlosses(ow_language);
            for (TranslatedContent tc : glosses)
            {
                System.out.println("Definiton: "+tc.getGloss());
            }
            //Retrieve the translation for all languages
            Set<SynTrans> translations = dm.getSynTranses();
            for (SynTrans st :translations)
            {
                System.out.println(OWLanguage.getName(st.getSyntrans().getLanguageId()) + " translation: "+ st.getSyntrans().getSpelling());
            }
            //Retrieve relations to other senses
            Map<DefinedMeaning,Integer> links = dm.getDefinedMeaningLinksAll();
            for (DefinedMeaning dm_target : links.keySet())
            {
                System.out.println(DefinedMeaningLinkType.getName(links.get(dm_target))+" relation with target "+ dm_target.getSpelling());
            }
        }

        System.exit(0);
    }
}